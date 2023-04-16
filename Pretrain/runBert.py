import argparse, random, pickle, torch, os, setproctitle
import numpy as np
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BertContrasPretrain import BertContrastive, BertContrastiveNeg
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup as warmup
from utilities import *
from file_dataset import ContrasDataset
from tqdm import tqdm
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--task",                       default="aol",          type=str,   help="<aol>|<tiangong>")
parser.add_argument("--per_gpu_batch_size",         default=128,            type=int)
parser.add_argument("--per_gpu_test_batch_size",    default=256,            type=int)
parser.add_argument("--learning_rate",              default=5e-5,           type=float, help="The initial learning rate for Adam.")
parser.add_argument("--scheduler",             action="store_true",    help="使用linear scheduler来递减学习率")
parser.add_argument("--warmup_step_rate",           default=0.1,            type=float, help="学习率warm up的比例")
parser.add_argument("--epochs",                     default=4,              type=int,   help="Total number of training epochs to perform.")
parser.add_argument("--save_path",                  default="./SCL/model/", type=str,   help="The path to save model.")
parser.add_argument("--bert_model_path",            default="./BERT/BERTModel/",type=str,help="The path to BERT model.")
parser.add_argument("--pretrain_model_path",        default="", type=str,   help="The path to load plm.")
parser.add_argument("--tqdm",                       action="store_true",    help="使用tqdm进度条")
parser.add_argument("--hint",                       type=str,               default="", help="模型提示，也是模型保存的目录文件名")
parser.add_argument('--seed',                       default=0,              type=int,   help="随机种子")
parser.add_argument("--log_path",                   default="./SCL/log/",   type=str,   help="The path to save log.")
parser.add_argument('--data_dir',                   default="./SCL/",       type=str,   help="数据存储目录")

parser.add_argument("--temperature",                default=0.1,            type=float, help="The temperature for CL.")
parser.add_argument('--need_better_result',         action="store_true",    help="只存储dev loss更小的模型")
parser.add_argument('--pkl_path',                   default="./SCL/data/aol",type=str,  help="pkl字典存储位置")
parser.add_argument('--use_hard_negatives',         default="True",         type=str,   help="对比学习预训练是否使用强负例<True>|<False>")
parser.add_argument('--del_ratio',                  default=0.6,            type=float,   help="随机删除的比例")
parser.add_argument('--rep_ratio',                  default=0.8,            type=float,   help="随机替换的比例")
args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
args.loss_path = os.path.join(args.save_path, "train_cl_loss.log")
args.param_logger = os.path.join(args.save_path, "params.log")
args.save_path = os.path.join(args.save_path, BertContrastive.__name__ + "." + args.task)   #! 如果使用的模型BertContrastive换了，则这里也要修改
setproctitle.setproctitle(args.hint)
logger = open(args.log_path, "a")
loss_logger = open(args.loss_path, "a")
param_logger = open(args.param_logger, "a")
device = torch.device("cuda:0")
args_dict = vars(args)
for arg in args_dict:
    print(arg, "==>", args_dict[arg], flush=True)
param_logger.write("\nHyper-parameters:\n")
param_logger.flush()
for k, v in args_dict.items():
    param_logger.write(str(k) + "\t" + str(v) + "\n")
    param_logger.flush()
seq_max_len = 128
if args.task == "aol":
    train_data = args.data_dir + "/aol/train/train_0101"
    test_data = args.data_dir + "/aol/dev/dev_0101"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 3
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
elif args.task == "tiangong":
    train_data = args.data_dir + "/tiangong/train/train.pos.txt"
    test_data = args.data_dir + "/tiangong/dev/dev.pos.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
else:
    assert False

#==========================#
def train_model(suffix):
    """
    在程序入口调用此函数，为对比预训练过程的入口
    suffix为读取数据文件名的后缀列表, [".route.txt", ".simrank.txt"]
    """
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    for index, suff in enumerate(suffix):
        if index == 0:
            # * 刚开始: 不加载模型或者加载use_pretrain_model指定的预训练模型
            print("load pre-train model from: %s" % (args.pretrain_model_path), flush=True)
            model_state_dict = torch.load(args.pretrain_model_path, map_location="cuda:0")
            bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
        else:
            # * 加载刚训练好的模型, 储存在args.save_path
            model_state_dict = torch.load(args.save_path, map_location="cuda:0")
            bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
        if args.use_hard_negatives == "True":  # 同时包含负样本
            model = BertContrastiveNeg(bert_model, args=args, temperature=args.temperature)    # 初始化模型类
        elif args.use_hard_negatives == "False":
            model = BertContrastive(bert_model, args=args, temperature=args.temperature)    # 初始化模型类
        n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print('* number of parameters: %d' % n_params, flush=True)
        model = model.to(torch.device("cuda:0"))
        model = torch.nn.DataParallel(model)
        fit(model, train_data, test_data, suff)

def fit(model, X_train: str, X_test: str, suff: str):
    """
    被train_model()调用
    构建数据集、优化器
    """
    train_dataset = ContrasDataset(
        args, X_train+suff, seq_max_len, tokenizer,
        deletion_ratio=args.del_ratio, replace_ratio=args.rep_ratio
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)  # step总数
    one_epoch_step = len(train_dataset) // args.batch_size  # 一个epoch内的step数量
    if args.scheduler_used: # 使用动态学习率优化
        scheduler = warmup(optimizer, num_warmup_steps=int(args.warmup_step_rate * one_epoch_step), num_training_steps=t_total)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = 1e4   # 预先设定一个loss较大值，每当loss减少时，更新此变量
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs, flush=True)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        logger.flush()
        loss_logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        loss_logger.flush()
        avg_loss = 0        # 每轮重新计算；计算所有step的平均loss
        model.train()
        if args.tqdm:
            epoch_iterator = tqdm(train_dataloader, ncols=120)
        else:
            epoch_iterator = train_dataloader
        for i, training_data in enumerate(epoch_iterator):  # i: 第i个step
            # 传入模型、batch数据、loss函数，
            # 计算训练集loss，与对比学习交叉熵预测的标签准确率
            loss, acc = train_step(model, training_data, bce_loss)  
            loss = loss.mean()
            acc = acc.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if args.scheduler_used:
                scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            if args.tqdm:
                epoch_iterator.set_postfix(lr=args.learning_rate, cont_loss=loss.item(), acc=acc.item())
            if i > 0 and i % 100 == 0:
                # 每100个step, 将当前loss写入loss_logger
                loss_logger.write("Step " + str(i) + ": " + str(loss.item()) + "\n")
                loss_logger.flush()
            if i > 0 and i % (one_epoch_step // 5) == 0:
                # 每次迭代进行20%整数倍后，在验证集上计算loss和acc
                best_result = evaluate(model, X_test, best_result, suff)
                model.train()
            avg_loss += loss.item()     # avg_loss加上每个step的loss
        # 以上: 一轮迭代结束
        cnt = one_epoch_step + 1    # 总step
        if args.tqdm:   # 计算此轮的平均loss
            tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        else:
            print("Average loss:{:.6f}".format(avg_loss / cnt), flush=True)
        # 迭代结束后再测一遍验证集
        best_result = evaluate(model, X_test, best_result, suff)
    #logger.close()
    #loss_logger.close()

def train_step(model, train_data, loss_func):
    """
    被fit()调用
    返回loss, acc
    """
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    if args.use_hard_negatives == 'True':
        sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg, sent_norm_neg = model.forward(train_data)
        # sent_rep: [bs, 768]; sent_norm: [bs, 1]
    elif args.use_hard_negatives == 'False':
        sent_rep1, sent_norm1, sent_rep2, sent_norm2 = model.forward(train_data)   
    else: assert False
    # 利用返回后的向量计算loss, acc
    if args.use_hard_negatives == 'True':
        contras_loss, acc = cal_loss_acc(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg, sent_norm_neg)
    elif args.use_hard_negatives == 'False':
        contras_loss, acc = cal_loss_acc(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2)
    else: assert False
    return contras_loss, acc

def evaluate(model, X_test, best_result, suff, is_test=False):
    """
    更新best_result
    被fit()调用
    对比预训练没有测试集! 因此is_test = False
    """
    y_test_loss, y_test_acc = predict(model, X_test, suff)
    # y_test_loss, y_test_acc 存储每一步的loss与acc
    result = np.mean(y_test_loss)
    y_test_acc = np.mean(y_test_acc)
    if args.need_better_result:
        if not is_test and result < best_result:    
            # 当前传入数据为验证集; 只要dev loss更小，就保存并更新best_result
            # 并且保存当前的参数
            best_result = result
            if args.tqdm:
                tqdm.write("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc))
            else:
                print("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc), flush=True)
            logger.write("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
            logger.flush()
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.save_path)
    else:   # 不需要检验dev loss
        if not is_test:    
            # 当前传入数据为验证集; 只要dev loss更小，就保存并更新best_result
            # 并且保存当前的参数
            best_result = result
            if args.tqdm:
                tqdm.write("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc))
            else:
                print("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc), flush=True)
            logger.write("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
            logger.flush()
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.save_path)
    return best_result

def predict(model, X_test, suff):
    """
    被evaluate()调用
    返回验证集loss, acc
    """
    model.eval()
    test_loss = []
    test_dataset = ContrasDataset(
        args, X_test+suff, seq_max_len, tokenizer,
        deletion_ratio=args.del_ratio, replace_ratio=args.rep_ratio)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, collate_fn=my_collate_fn)
    y_test_loss = []
    y_test_acc = []
    with torch.no_grad():
        if args.tqdm:
            epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        else:
            epoch_iterator = test_dataloader
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            # sent_rep: [bs, 768]; sent_norm: [bs, 1]
            if args.use_hard_negatives == 'True':   
                # 包含负样本
                sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg, sent_norm_neg = model.forward(test_data)
            elif args.use_hard_negatives == 'False':
                # 不含负样本
                sent_rep1, sent_norm1, sent_rep2, sent_norm2 = model.forward(test_data)   
            else: assert False

            # 利用返回后的向量计算loss, acc
            if args.use_hard_negatives == 'True':
                contras_loss, acc = cal_loss_acc(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2, sent_rep_neg, sent_norm_neg)
            elif args.use_hard_negatives == 'False':
                contras_loss, acc = cal_loss_acc(args, sent_rep1, sent_norm1, sent_rep2, sent_norm2)
            else: assert False
            test_loss = contras_loss.mean()
            test_acc = acc.mean()
            # 存储每一步的loss, acc
            y_test_loss.append(test_loss.item())
            y_test_acc.append(test_acc.item())
    y_test_loss = np.asarray(y_test_loss)
    y_test_acc = np.asarray(y_test_acc)
    return y_test_loss, y_test_acc


if __name__ == '__main__':
    set_seed(args.seed)
    #filename = [".simrank.txt", ".route.txt", "trans.txt"]
    filename = [".trans.route.simrank.txt"]
    train_model(filename)
    logger.close()
    loss_logger.close()
    param_logger.close()

    