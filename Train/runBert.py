import random, torch, transformers, os, pytrec_eval
import numpy as np
from BertSessionSearch import BertSessionSearch
from dataclasses import dataclass, field
from typing import Optional
from transformers import BertTokenizer, BertModel, Trainer, BertConfig, TrainingArguments
from file_dataset import FileDataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/yutao_zhu/BertModel/")
    tokenizer_name: Optional[str] = field(default="/home/yutao_zhu/BertModel/")
    load_plm: Optional[str] = field(default='/home/zhaoheng_huang/CAGS_result/SCL/CLModel/BertContrastive.aol')
    load_model: Optional[str] = field(default='/home/zhaoheng_huang/CAGS_result/Ranking/ROUTE/PointBertSessionSearch.aol')

@dataclass
class DataArguments:
    data_path: str = field(default="./data/aol/", metadata={"help": "Path to the data."})
    task: str = field(default="aol")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    one_sess = []
    sessions = []
    for one_pred, one_label in zip(preds, labels):
        one_sess.append([one_pred, one_label])
        if len(one_sess) % 50 == 0:
            one_sess_tmp = np.array(one_sess)
            if one_sess_tmp[:, 1].sum() > 0:
                sessions.append(one_sess)
            one_sess = []
    qrels = {}
    run = {}
    for idx, sess in enumerate(sessions):
        query_id = str(idx)
        if query_id not in qrels:
            qrels[query_id] = {}
        if query_id not in run:
            run[query_id] = {}
        for jdx, r in enumerate(sess):
            doc_id = str(jdx)
            qrels[query_id][doc_id] = int(r[1])
            run[query_id][doc_id] = float(r[0])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recip_rank', 'ndcg_cut.1,3,5,10'})
    res = evaluator.evaluate(run)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    ndcg_1_list = [v['ndcg_cut_1'] for v in res.values()]
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res.values()]
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]
    return {
        "map": np.average(map_list),
        "mrr": np.average(mrr_list),
        "ndcg@1": np.average(ndcg_1_list),
        "ndcg@3": np.average(ndcg_3_list),
        "ndcg@5": np.average(ndcg_5_list),
        "ndcg@10": np.average(ndcg_10_list)
    } 

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def train_model(model_args, data_args, training_args):
    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name)
    if data_args.task == "aol":
        train_data = data_args.data_path + "/train/train_line.txt"
        test_data = data_args.data_path + "/test/test_line.middle.txt"
        predict_data = data_args.data_path + "/test/test_line.txt"
        additional_tokens = 3
        tokenizer.add_tokens("[eos]")
        tokenizer.add_tokens("[term_del]")
        tokenizer.add_tokens("[sent_del]")
    elif data_args.task == "tiangong":
        train_data = "./data/tiangong/train.point.txt"
        test_last_data = "./data/tiangong/test.point.lastq.txt"
        test_pre_data = "./data/tiangong/test.point.preq.txt"
        predict_data = "./data/tiangong/test.txt"
        additional_tokens = 2
        tokenizer.add_tokens("[eos]")
        tokenizer.add_tokens("[empty_d]")
    else:
        assert False

    # load model
    config = BertConfig.from_pretrained(model_args.model_name_or_path)
    bert_model = BertModel.from_pretrained(model_args.model_name_or_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(model_args.load_plm)
    bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
    model = BertSessionSearch(bert_model, config)
    model.bert_model.gradient_checkpointing_enable()

    train_dataset = FileDataset(train_data, 128, tokenizer)
    test_dataset = FileDataset(test_data, 128, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    set_seed()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train_model(model_args, data_args, training_args)