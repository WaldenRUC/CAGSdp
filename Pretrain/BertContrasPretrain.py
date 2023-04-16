import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
hidden_size = 768
class BertContrastive(nn.Module):
    """
    不含负样本, 传统对比学习
    """
    def __init__(self, bert_model, temperature, args):
        super(BertContrastive, self).__init__()
        self.bert_model = bert_model
        self.temperature = temperature
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')  # 返回一个和bsz大小相等的vector，而非一个数
        self.args = args
    def forward(self, batch_data):
        batch_size = batch_data["input_ids1"].size(0)
        #==========#
        input_ids1 = batch_data["input_ids1"]
        attention_mask1 = batch_data["attention_mask1"]
        token_type_ids1 = batch_data["token_type_ids1"]
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        sent_rep1 = self.bert_model(**bert_inputs1)[1]          # pooler output: [bsz, 768]
        assert sent_rep1.shape[0] == batch_size and sent_rep1.shape[1] == hidden_size
        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # sqrt(x1^2 + x2^2 + ...)   # [bsz, 1]
        assert sent_norm1.shape[0] == batch_size and sent_norm1.shape[1] == 1
        #==========#
        input_ids2 = batch_data["input_ids2"]
        attention_mask2 = batch_data["attention_mask2"]
        token_type_ids2 = batch_data["token_type_ids2"]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        sent_rep2 = self.bert_model(**bert_inputs2)[1]
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)
        #==========#
        return sent_rep1, sent_norm1, sent_rep2, sent_norm2




class BertContrastiveNeg(nn.Module):
    """
    含强负样本的对比学习
    """
    def __init__(self, bert_model, temperature, args):
        super(BertContrastiveNeg, self).__init__()
        self.bert_model = bert_model
        self.temperature = temperature
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')  # 返回一个和bsz大小相等的vector，而非一个数
        self.args = args
    def forward(self, batch_data):
        batch_size = batch_data["input_ids1"].size(0)
        #==========#
        input_ids1 = batch_data["input_ids1"]
        attention_mask1 = batch_data["attention_mask1"]
        token_type_ids1 = batch_data["token_type_ids1"]
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        sent_rep1 = self.bert_model(**bert_inputs1)[1]          # pooler output: [bsz, 768]
        assert sent_rep1.shape[0] == batch_size and sent_rep1.shape[1] == hidden_size
        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # sqrt(x1^2 + x2^2 + ...)   # [bsz, 1]
        assert sent_norm1.shape[0] == batch_size and sent_norm1.shape[1] == 1
        #==========#
        input_ids2 = batch_data["input_ids2"]
        attention_mask2 = batch_data["attention_mask2"]
        token_type_ids2 = batch_data["token_type_ids2"]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        sent_rep2 = self.bert_model(**bert_inputs2)[1]
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)
        #==========#
        input_ids_neg = batch_data["input_ids_neg"]
        token_type_ids_neg = batch_data["token_type_ids_neg"]
        attention_mask_neg = batch_data["attention_mask_neg"]
        bert_inputs3 = {'input_ids': input_ids_neg, 'attention_mask': attention_mask_neg, 'token_type_ids': token_type_ids_neg}
        sent_rep3 = self.bert_model(**bert_inputs3)[1]
        sent_norm3 = sent_rep3.norm(dim=-1, keepdim=True)
        #==========#
        return sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep3, sent_norm3