import torch.nn as nn
import torch.nn.init as init
import torch
from transformers import BertTokenizer
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

class BertSessionSearchList(nn.Module):
    def __init__(self, bert_model, config):
        super(BertSessionSearchList, self).__init__()
        self.bert_model = bert_model
        self.config = config
        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """
        Args:
            input_ids ([bsz, 5, seq_len]): [description]
            attention_mask ([bsz, 5, seq_len]): [description]
            token_type_ids ([bsz, 5, seq_len]): [description]
            labels ([bsz])
        """
        bsz, ce_width, seq_len = input_ids.shape
        bert_inputs = {'input_ids': input_ids.view(bsz*ce_width, -1), 'attention_mask': attention_mask.view(bsz*ce_width, -1), 'token_type_ids': token_type_ids.view(bsz*ce_width, -1)}
        bert_outputs = self.bert_model(**bert_inputs)
        sent_rep = self.dropout(bert_outputs[1]).view(bsz, ce_width, -1)    # [bsz*ce_width, 768] -> [bsz, ce_width, 768]
        logits = self.classifier(sent_rep)      # [bsz, ce_width] ?
        loss = self.ce_loss(logits, labels)   

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )


class BertSessionSearch(nn.Module):
    def __init__(self, bert_model, config):
        super(BertSessionSearch, self).__init__()
        self.bert_model = bert_model
        self.config = config
        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """
        Args:
            input_ids ([bsz, 5, seq_len]): [description]
            attention_mask ([bsz, 5, seq_len]): [description]
            token_type_ids ([bsz, 5, seq_len]): [description]
        """
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        bert_outputs = self.bert_model(**bert_inputs)
        sent_rep = self.dropout(bert_outputs[1])
        logits = self.classifier(sent_rep)
        loss = self.bce_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
