#source /data/shuting_wang/anaconda3/bin/activate /data/shuting_wang/cags
source /home/zhaoheng_huang/anaconda3/bin/activate /home/zhaoheng_huang/anaconda3/envs/cagsdp
CODE=/home/zhaoheng_huang/CAGSdp
DATA=/home/zhaoheng_huang/CAGS_data
RES=/home/zhaoheng_huang/CAGS_result
host=151

# 以下不需修改
curTime="$(date +%y-%m-%d=%H:%m)"
Seed=0
dataDir=${DATA}/Pretrain/data        # Rank所需的数据目录
saveDir=${RES}/Pretrain/${curTime}   # 模型日志保存的目录
mkdir -p ${saveDir}                 # 创建这个目录
# 以上不需修改

outputLogPath=${saveDir}/pretrain.log   # 以增量方式写入日志
echo ${outputLogPath}
hint="simple_pretrain"
plm=${RES}/BERT/BERTModel/pytorch_model.bin              # 从BERT开始pretrain
#plm=${RES}/SCL/CLModel/BertContrastive.aol          # 加载一个预训练模型，继续预训练
visible_devices="0,1"     # 指定可视GPU后，用所有的卡

CUDA_VISIBLE_DEVICES=$visible_devices python -u ./Pretrain/runBert.py \
    --task aol \
    --per_gpu_batch_size 160 \
    --per_gpu_test_batch_size 320 \
    --learning_rate 5e-5 \
    --scheduler \
    --warmup_step_rate 0 \
    --epochs 3 \
    --save_path ${saveDir} \
    --bert_model_path ${RES}/BERT/BERTModel \
    --pretrain_model_path ${plm} \
    --tqdm \
    --hint ${hint} \
    --seed ${Seed} \
    --log_path ${outputLogPath} \
    --data_dir ${dataDir} > ${outputLogPath} 2>&1

python -u ${CODE}/Notify/notify_fs.py \
    --text "${host}训练完毕【$hint】"