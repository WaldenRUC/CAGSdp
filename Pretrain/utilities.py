import pickle, os, random
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import torch.nn.functional as F
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def load_pkl(args, pkl_path: str, filelist: list):
    """
    加载预处理的pkl文件
    pkl_dict = load_pkl(args, "<path>", ["q2cod.pkl", ...])
    """
    res_dict = {}
    if args.tqdm: filelist = tqdm(filelist)
    for file in filelist:   # file: q2coq_deep_train, etc.
        with open(os.path.join(pkl_path, file+".pkl"), 'rb') as fp:
            res_dict[file] = pickle.load(fp)
    return res_dict
def my_collate_fn(batch):
    """
    过滤为None的数据(见BertContrasPretrain.py)
    """
    # batch: [Dict, None, Dict, Dict, ...] ==> [Dict, Dict, ...]
    batch = [item for item in batch if item != None]
    if len(batch) == 0: # all elements filtered out
        return torch.Tensor()
    batch = default_collate(batch)
    return batch

def cal_loss_acc(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg=None, sent_norm_neg=None):
    # sent_rep: [bs, 768]; sent_norm: [bs, 1]
    # sent_rep1 & sent_rep2 对应第一维下标的两个向量为正例
    # 矩阵相乘得到的[bs, bs]矩阵中，对角上的是正例
    # * 负样本为其他N-1个，而非2(N-1)个
    """
    batch:
    s11     s12
    ...     ...
    sn1     sn2
    ---
    self_11算的是s11~sn1这一列的相似度矩阵
    cross_12算的是s11~sn1与s12~sn2的相似度矩阵
    self_22算的是s12~sn2这一列的相似度矩阵
    cross_21算的是s12~sn2与s11~sn1的相似度矩阵
    """
    # 返回时collect所有卡的scalar，并回到主卡device:0上
    size = sent_rep1.shape[0]
    batch_arange = torch.arange(size).to("cuda:%s"%(args.device_id))
    if sent_rep_neg == None:    # 无负样本
        mask = F.one_hot(batch_arange, num_classes=size + 1) * -1e10
    else:   # 有负样本，多一列
        mask = F.one_hot(batch_arange, num_classes=size + 2) * -1e10
    """
    mask (including neg): (size = 4)
    [-1e10, 0, 0, 0, 0, 0],
    [0, -1e10, 0, 0, 0, 0],
    [0, 0, -1e10, 0, 0, 0],
    [0, 0, 0, -1e10, 0, 0]
    """
    # 获得第1 ~ size列的数据
    batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)    # [batch, batch]
    batch_self_11 = batch_self_11 / args.temperature

    # 获得第size+1列的数据(正例)
    batch_pos = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]
    batch_pos = batch_pos / args.temperature
    batch_pos = torch.diagonal(batch_pos)   # [batch], 保留对角线(正例)
    batch_pos = torch.unsqueeze(batch_pos, dim=-1)  # [batch, 1]
    
    # 获得第size+2列的数据(负例)
    if sent_rep_neg == None:    # 无负样本
        # [batch]
        batch_label1 = torch.full([size], size).to("cuda:%s"%(args.device_id))   # [4, 4, 4, 4]
        # [batch, batch + 1]
        batch_res1 = torch.cat([batch_self_11, batch_pos], dim=-1)  # [batch, batch + 1]
        batch_res1 += mask  # 加了mask后, logit不会识别出自身
        cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        contras_loss = cl_loss(batch_res1, batch_label1)
        batch_logit = batch_res1.argmax(dim = -1)
        acc = torch.sum(batch_logit == batch_label1).float() / size
    else:   # 有负样本
        # [batch]
        batch_label1 = torch.full([size], size).to("cuda:%s"%(args.device_id))   # [4, 4, 4, 4]
        batch_neg = torch.einsum("ad,bd->ab", sent_rep1, sent_rep_neg) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm_neg) + 1e-6)  # [batch, batch]
        batch_neg = batch_neg / args.temperature
        batch_neg = torch.diagonal(batch_neg)   # [batch]
        batch_neg = torch.unsqueeze(batch_neg, dim=-1)  # [batch, 1]
        # [batch, batch + 2]
        batch_res1 = torch.cat([batch_self_11, batch_pos, batch_neg], dim=-1)  # [batch, batch + 2]
        batch_res1 += mask  # 加了mask后, logit不会识别出自身
        # TODO 这里的cl_loss给强负例(最后一列)加一个weight
        cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        contras_loss = cl_loss(batch_res1, batch_label1)
        batch_logit = batch_res1.argmax(dim = -1)
        acc = torch.sum(batch_logit == batch_label1).float() / size
    return contras_loss, acc
        
    

def cal_loss_acc_0102(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg=None, sent_norm_neg=None):
    # sent_rep: [bs, 768]; sent_norm: [bs, 1]
    # sent_rep1 & sent_rep2 对应第一维下标的两个向量为正例
    # 矩阵相乘得到的[bs, bs]矩阵中，对角上的是正例
    """
    batch:
    s11     s12
    ...     ...
    sn1     sn2
    ---
    self_11算的是s11~sn1这一列的相似度矩阵
    cross_12算的是s11~sn1与s12~sn2的相似度矩阵
    self_22算的是s12~sn2这一列的相似度矩阵
    cross_21算的是s12~sn2与s11~sn1的相似度矩阵
    """
    # 返回时collect所有卡的scalar，并回到主卡device:0上
    size = sent_rep1.shape[0]
    batch_arange = torch.arange(size).to("cuda:%s"%(args.device_id))
    if sent_rep_neg == None:    # 无负样本
        mask = F.one_hot(batch_arange, num_classes=size * 2) * -1e10
    else:   # 有负样本，多一列
        mask = F.one_hot(batch_arange, num_classes=size * 2 + 1) * -1e10
    """
    mask (including neg): (size = 4)
    [-1e10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1e10, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1e10, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1e10, 0, 0, 0, 0, 0]
    """
    # 获得第1 ~ size, size+1 ~ 2*size列的数据
    batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)
    batch_self_11 = batch_self_11 / args.temperature
    batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]
    batch_cross_12 = batch_cross_12 / args.temperature
    
    # 和neg交互，生成最后一列
    if sent_rep_neg == None:    # 无负样本
        # [size]
        batch_label1 = batch_arange + size  # [0, 1, 2, 3] ==> [4, 5, 6, 7]
        # [size, size * 2]
        batch_res1 = torch.cat([batch_self_11, batch_cross_12], dim=-1)  
        batch_res1 += mask  # 加了mask后, logit不会识别出自身
        cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        contras_loss = cl_loss(batch_res1, batch_label1)
        batch_logit = batch_res1.argmax(dim = -1)
        acc = torch.sum(batch_logit == batch_label1).float() / size
    else:   # 有负样本
        batch_cross_13 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep_neg) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm_neg) + 1e-6)
        batch_cross_13 = torch.diagonal(batch_cross_13) # [size]
        batch_cross_13 = torch.unsqueeze(batch_cross_13, dim=-1)    #[size, 1]
        batch_cross_13 = batch_cross_13 / args.temperature
        # [size]
        batch_label1 = batch_arange + size  # [0, 1, 2, 3] ==> [4, 5, 6, 7]
        # [size, size * 2 + 1]
        batch_res1 = torch.cat([batch_self_11, batch_cross_12, batch_cross_13], dim=-1)
        batch_res1 += mask
        # TODO 这里的cl_loss给强负例(最后一列)加一个weight
        cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        contras_loss = cl_loss(batch_res1, batch_label1)
        batch_logit = batch_res1.argmax(dim = -1)
        acc = torch.sum(batch_logit == batch_label1).float() / size
    return contras_loss, acc

