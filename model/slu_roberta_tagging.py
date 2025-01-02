#coding=utf8

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.initialization import set_torch_device

class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob,)


class SLURoberta(nn.Module):

    def __init__(self, config):
        super(SLURoberta, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.device = set_torch_device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext",)
        self.model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                               output_hidden_states=True).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        # RoBERTa hidden size
        self.hidden_size = 768
        self.output_layer = TaggingFNNDecoder(self.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask

        self.length = [len(utt) for utt in batch.utt]
        encoded_inputs = self.tokenizer(batch.utt,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=max(self.length),
                                        return_tensors='pt').to(self.device)
        hidden_states = self.model(**encoded_inputs).hidden_states
        hiddens = hidden_states[-1]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
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
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()