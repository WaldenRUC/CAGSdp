import torch
import linecache
from torch.utils.data import Dataset
import numpy as np
import random
class FileDatasetList(Dataset):
    """
    list-wise learning所需数据集
    文件读取的每一行: <seq1_click>[####]<seq1_unclick>[####]<seq2_unclick>[####]...
    """
    def __init__(self, filename, max_seq_length, tokenizer):
        super(FileDatasetList, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        with open(filename, 'r') as f:
            self._total_data = len(f.readlines())
    def check_length(self, pairlist):
        assert len(pairlist) % 2 == 0
        q_d_minimum_length = 0
        for i in range(len(pairlist)):
            q_d_minimum_length += len(pairlist[i]) + 1
        max_seq_length = self._max_seq_length - 3
        
        if q_d_minimum_length > max_seq_length:
            pairlist.pop(0)
            pairlist.pop(0)
            pairlist = self.check_length(pairlist)
        return pairlist
    def anno_main(self, qd_pairs):
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([1] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx+1)
        line = line.strip().split("[####]")
        assert len(line) == 5
        input_ids_list, segment_ids_list, attention_mask_list = [], [], []
        for seq in line:
            input_ids, attention_mask, segment_ids = self.anno_main(seq.split("\t"))
            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            attention_mask_list.append(attention_mask)
        batch = {
            "input_ids": np.asarray(input_ids_list),
            "token_type_ids": np.asarray(segment_ids_list),
            "attention_mask": np.asarray(attention_mask_list),
            "label": torch.LongTensor([0]) # 第一个是点击的序列
        }
        return batch
    def __len__(self):
        return self._total_data

class FileDatasetContras(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer):
        super(FileDatasetContras, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    def check_length(self, pairlist):
        assert len(pairlist) % 2 == 0
        # TODO
    def __getitem__(self, idx):
        pass
        # TODO
    def __len__(self):
        return self._total_data

class FileDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        assert len(pairlist) % 2 == 0
        q_d_minimum_length = 0
        for i in range(len(pairlist)):
            q_d_minimum_length += len(pairlist[i]) + 1
        max_seq_length = self._max_seq_length - 3
        
        if q_d_minimum_length > max_seq_length:
            pairlist.pop(0)
            pairlist.pop(0)
            pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_main(self, qd_pairs):
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([1] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = float(line[0])
        qd_pairs = line[1:]
        input_ids, attention_mask, segment_ids = self.anno_main(qd_pairs)
        batch = {
            'input_ids': input_ids, 
            'token_type_ids': segment_ids, 
            'attention_mask': attention_mask, 
            'label': np.asarray([label])
        }
        return batch
    
    def __len__(self):
        return self._total_data

