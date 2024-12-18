import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_tags):
        super(PointerNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.pointer = nn.Linear(hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=args.tag_pad_idx)

    def forward(self, input_ids, tag_ids=None):
        embedded = self.embedding(input_ids)
        encoder_outputs, _ = self.encoder_lstm(embedded)
        decoder_outputs, _ = self.decoder_lstm(encoder_outputs)
        logits = self.pointer(decoder_outputs)

        if tag_ids is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), tag_ids.view(-1))
            return logits, loss
        return (logits,)

# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam
from transformers import BertTokenizer
from typing import List, Tuple

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args_pointer_network import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from tqdm.auto import tqdm
import torch
import torch.nn as nn

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

tokenizer = BertTokenizer.from_pretrained(args.bert_path)
model = PointerNet(vocab_size=tokenizer.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_tags=args.num_tags).to(device)
Example.word2vec.load_embeddings(model.embedding, Example.word_vocab, device=device)

if args.testing:
    check_point = torch.load(open('model_pointer.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer

def prepare_input(args, cur_dataset: List[Example], tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    tag_ids = []
    attn_masks = []

    max_len = max([len(example.utt) for example in cur_dataset])

    for example in cur_dataset:
        encoded_dict = tokenizer(
            example.utt,
            add_special_tokens=True, 
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids.append(encoded_dict['input_ids'])
        tags = example.tag_id + [args.tag_pad_idx] * (max_len - len(example.tag_id))
        tag_ids.append(torch.tensor(tags, dtype=torch.long).unsqueeze(0))
        attn_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    tag_ids = torch.cat(tag_ids, dim=0).to(device)
    attn_masks = torch.cat(attn_masks, dim=0).to(device)

    return input_ids, tag_ids, attn_masks

def tags_to_triples(preds: List[List], examples: List[Example]):
    batch_size = len(examples)
    predictions = []
    for i in range(batch_size):
        pred, example = preds[i], examples[i]
        utt = example.utt
        pred = pred[:len(utt)]
        pred_tuple = []
        idx_buff, tag_buff, pred_tags = [], [], []
        for idx, tid in enumerate(pred):
            tag = Example.label_vocab.convert_idx_to_tag(tid)
            pred_tags.append(tag)
            if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([utt[j] for j in idx_buff])
                idx_buff, tag_buff = [], []
                pred_tuple.append(f'{slot}-{value}')
                if tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            elif tag.startswith('I') or tag.startswith('B'):
                idx_buff.append(idx)
                tag_buff.append(tag)
        if len(tag_buff) > 0:
            slot = '-'.join(tag_buff[0].split('-')[1:])
            value = ''.join([utt[j] for j in idx_buff])
            pred_tuple.append(f'{slot}-{value}')
        predictions.append(pred_tuple)
    return predictions

def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]

            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, tokenizer, device)
            output = model(input_ids, tag_ids=tag_ids)
            prob, loss = output[0], output[1]

            pred = torch.argmax(prob, dim=-1).cpu().tolist()
            tag_ids = tag_ids.cpu().tolist()
            pred, label = tags_to_triples(pred, cur_dataset), tags_to_triples(tag_ids, cur_dataset)
            predictions.extend(pred)
            labels.extend(label)

            total_loss += loss.item()
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, tokenizer, device)

            logits = model(input_ids)[0]
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            pred = tags_to_triples(pred, cur_dataset)
            for pi, p in enumerate(pred):
                did = cur_dataset[pi].did
                predictions[did] = p
    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):

            # cur_dataset: list of Example objects
            cur_dataset: List["Example"] = [train_dataset[k] for k in train_index[j: j + step_size]]
            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, tokenizer, device)
            logits, loss = model(input_ids, tag_ids=tag_ids)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model_pointer_network.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))