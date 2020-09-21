#!/bin/bash
#SBATCH --job-name=gpu-1nodes-fr
#SBATCH --partition=gpu_p1
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=slurm_files/Travail-1gpu-fr-%j.out
#SBATCH --error=slurm_files/Travail-1gpu-fr-%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

module purge

set -x

LANGUAGE='fr'
TASK='denoising'
DATA_PATH='../../pretraining_data/data-bin-fr'
MODEL_PATH='small_bart.pt'
MAX_SENTENCES=64
MAX_TOKENS=6000
MAX_UPDATE=105000
SAVE_INTERVAL=5000
LR=0.0004
MAX_EPOCH=20
DISTRIBUTED_WORLD_SIZE=1
SENTENCE_PIECE_MODEL='../../sentence_piece_multilingual.model'
VALID_SUBSET='valid'

TENSORBOARD_LOGS=../../tensorboard_logs/$task/$LANGUAGE/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_si${SAVE_INTERVAL}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}
SAVE_DIR=../../checkpoints/$task/$LANGUAGE/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_si${SAVE_INTERVAL}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}

mkdir -p $SAVE_DIR

CUDA_VISIBLE_DEVICES=0

srun fairseq-train $DATA_PATH \
    --optimizer=adam \
    --adam-betas='(0.9, 0.999)' \
    --adam-eps=1e-06 \
    --arch='bart_small' \
    --bpe='sentencepiece' \
    --sentencepiece-vocab $SENTENCE_PIECE_MODEL \
    --clip-norm=0.1 \
    --log-interval=10 \
    --mask=0.3 \
    --mask-length='span-poisson' \
    --mask-random=0.1 \
    --permute-sentences=1 \
    --poisson-lambda=3.5 \
    --replace-length=1 \
    --rotate=0 \
    --max-update $MAX_UPDATE \
    --total-num-update $MAX_UPDATE \
    --save-dir $SAVE_DIR \
    --save-interval-updates=$SAVE_INTERVAL \
    --skip-invalid-size-inputs-valid-test \
    --task='denoising' \
    --update-freq=8 \
    --restore-file=$MODEL_PATH \
    --required-batch-size-multiple 8 \
    --fp16 \
    --max-sentences $MAX_SENTENCES \
    --lr=$LR \
    --weight-decay=0.01 \
    --lr-scheduler polynomial_decay \
    --activation-fn 'gelu' \
    --pooler-activation-fn 'tanh' \
    --tensorboard-logdir=$TENSORBOARD_LOGS \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --reset-lr-scheduler \
    --max-tokens=$MAX_TOKENS \
    --dropout 0.1 \
    --dataset-impl 'mmap' \
    --max-epoch $MAX_EPOCH \
    --warmup-updates $((6*$MAX_UPDATE/100)) \
    --no-epoch-checkpoints \
    --valid-subset $VALID_SUBSET
