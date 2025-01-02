#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import random


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = nn.LSTM(config.embed_size,
                           config.hidden_size // 2,
                           num_layers=config.num_layer,
                           bidirectional=True,
                           batch_first=True)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, batch):
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.embedding(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout(rnn_out)

        return hiddens

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)

        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=-1)
        
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(-1)
        attention_weights = torch.softmax(attention, dim=1)
        attention_weights = attention_weights.unsqueeze(1)

        context_vector = attention_weights @ encoder_outputs
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights
        

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.num_tags = config.num_tags
        self.hidden_size = config.hidden_size
        self.pad_idx = config.tag_pad_idx
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=16, dropout=config.dropout)
        self.output_lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True, bidirectional=False)
        self.tagging = nn.Linear(self.hidden_size, self.num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, encoder_outputs, mask, labels=None):
        batch_size, seq_len, _ = encoder_outputs.size()

        decoder_hidden = torch.zeros(batch_size, self.hidden_size).to(encoder_outputs.device)
        all_logits = []
        all_probs = []

        for t in range(seq_len):
            Q = encoder_outputs.transpose(0, 1)[t].unsqueeze(0)
            KV = encoder_outputs.transpose(0, 1)
            attention_output, _ = self.attention(Q, KV, KV, key_padding_mask=mask)
            context_vector = attention_output.transpose(0, 1).squeeze(1) # [batch_size, hidden_size]

            if labels is not None:
                if random.random() < 0.5:
                    decoder_input = torch.cat((context_vector, labels[:, t].unsqueeze(1).expand(-1, self.hidden_size)), dim=-1)
                else:
                    decoder_input = torch.cat((context_vector, decoder_hidden), dim=-1)
            else:
                decoder_input = torch.cat((context_vector, decoder_hidden), dim=-1)

            lstm_output, _ = self.output_lstm(decoder_input)
            decoder_output = self.dropout(lstm_output)

            logits = self.tagging(decoder_output)
            logits += (1 - mask[:, t].unsqueeze(-1).repeat(1, self.num_tags)) * -1e32

            prob = torch.softmax(logits, dim=-1)

            all_logits.append(logits.unsqueeze(1))
            all_probs.append(prob.unsqueeze(1))

            decoder_hidden = decoder_output

        all_logits = torch.cat(all_logits, dim=1)
        all_probs = torch.cat(all_probs, dim=1)

        if labels is not None:
            loss = self.loss_fct(all_logits.view(-1, all_logits.shape[-1]), labels.view(-1))
            return all_probs, loss
        return (all_probs, )


class SLUSeq2Seq(nn.Module):
    def __init__(self, config):
        super(SLUSeq2Seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, batch):
        hiddens = self.encoder(batch)
        tag_output = self.decoder(hiddens, batch.tag_mask, batch.tag_ids)
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
