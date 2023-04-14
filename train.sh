source /home/zhaoheng_huang/anaconda3/bin/activate /home/zhaoheng_huang/anaconda3/envs/cagsdp
CODE=/home/zhaoheng_huang/CAGSdp
DATA=/home/zhaoheng_huang/CAGS_data
RES=/home/zhaoheng_huang/CAGS_result
host=151

# 以下不需修改
curTime="$(date +%y-%m-%d=%H:%m)"
Seed=0
dataDir=${DATA}/Rank/data        # Rank所需的数据目录
saveDir=${RES}/Rank/${curTime}   # 模型日志保存的目录
mkdir -p ${saveDir}                 # 创建这个目录
# 以上不需修改


deepspeed --include localhost:0,1 runBert.py \
  --model_name_or_path ${RES}/BERT/BERTModel \
  --tokenizer_name ${RES}/BERT/BERTModel \
  --data_path ${DATA}/Rank/data/aol \
  --task aol \
  --load_plm ${RES}/SCL/CLModel/BertContrastive.aol \
  --load_model ${RES}/Ranking/ROUTE/PointBertSessionSearch.aol \
  --output_dir ${RES}/Rank \
  --optim adamw_torch \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2048 \
  --per_device_eval_batch_size 2048 \
  --evaluation_strategy steps \
  --logging_steps 100 \
  --log_level warning \
  --load_best_model_at_end True \
  --metric_for_best_model map \
  --greater_is_better True \
  --save_total_limit 1 \
  --eval_steps 100 \
  --save_steps 100 \
  --fp16 True \
  --deepspeed dp.json