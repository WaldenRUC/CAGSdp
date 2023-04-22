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

CUDA_VISIBLE_DEVICES=1 python -u ./transE/run.py \
    --epochs 50000 \
    --batch_size 128