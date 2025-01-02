# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam, AdamW

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args_transformer import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from typing import List
from tqdm.auto import tqdm
from transformers import BertTokenizer
from torch import nn
from torch.nn import Transformer
import torch
from typing import List, Tuple

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

args.vocab_size = Example.word_vocab.vocab_size # Vocab: Mapping word to index (int), 0 = <pad>, 1 = <unk>
args.pad_idx = Example.word_vocab[PAD] # index of <pad> in word2idx is 0
args.num_tags = Example.label_vocab.num_tags 
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

tokenizer = BertTokenizer.from_pretrained(args.bert_path)

args.vocab_size = tokenizer.vocab_size


import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # pe = torch.zeros(max_len, embed_dim)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)

        pe = nn.Parameter(torch.randn(max_len, embed_dim))
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        # self.positional_encoding = PositionalEncoding(config.embed_size)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config.embed_size, config.num_heads, dropout=config.dropout) for _ in range(config.num_blocks)]
        )
        self.linear = nn.Linear(config.embed_size, config.num_tags)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        # x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Transformer expects input of shape (seq_len, batch_size, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, embed_dim)
        logits = self.linear(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return logits, loss
        return logits

model = TransformerModel(args).to(device)

# Example.word2vec.load_embeddings(model.embedding, Example.word_vocab, device=device)

if args.testing:
    check_point = torch.load(open('model_transformer.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def prepare_input(args, cur_dataset: List[Example], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ex_list = sorted(cur_dataset, key=lambda x: len(x.input_idx), reverse=True)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    utt = [ex.utt for ex in ex_list]
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    lengths = input_lens
    did = [ex.did for ex in ex_list]

    labels = [ex.slotvalue for ex in ex_list]
    tag_lens = [len(ex.tag_id) for ex in ex_list]
    max_tag_lens = max(tag_lens)
    tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
    tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
    tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
    tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)

    return input_ids, tag_ids, tag_mask

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

            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, device)
            logits, loss = model(input_ids, labels=tag_ids)

            pred = torch.argmax(logits, dim=-1).cpu().tolist()
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
            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, device)

            logits = model(input_ids)
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
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            input_ids, tag_ids, attn_masks = prepare_input(args, cur_dataset, device)
            logits, loss = model(input_ids, labels=tag_ids)

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
            }, open('model_transformer.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))