DATA_SET='cls-music'
MODEL='M_fr'
TASK='sentence_prediction'
DATA_PATH='../../../FLUE_data/data-FLUE-CLS-music'
MODEL_PATH='../../../checkpoints/denoising/fr/ms64_mu105000_si5000_lr0.0004_me20_dws4/checkpoint_last.pt'
MAX_SENTENCES=32
MAX_UPDATE=1250
LR=1e-05
MAX_EPOCH=25
DISTRIBUTED_WORLD_SIZE=1
SENTENCE_PIECE_MODEL='../../../sentence_piece_multilingual.model'
VALID_SUBSET='valid,test'
NUM_CLASSES=2
SEED=$1

TENSORBOARD_LOGS=../../../tensorboard_logs/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}/$SEED
SAVE_DIR=../../../checkpoints/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}/$SEED

CUDA_VISIBLE_DEVICES=0

fairseq-train $DATA_PATH \
    --restore-file $MODEL_PATH \
    --max-sentences $MAX_SENTENCES \
    --task $TASK \
    --update-freq 1 \
    --seed $SEED \
    --add-prev-output-tokens \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --separator-token 2 \
    --arch bart_small \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --find-unused-parameters \
    --bpe 'sentencepiece' \
    --sentencepiece-vocab $SENTENCE_PIECE_MODEL \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric 'accuracy' \
    --no-save \
    --save-dir $SAVE_DIR \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --max-update $MAX_UPDATE \
    --total-num-update $MAX_UPDATE \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --tensorboard-logdir $TENSORBOARD_LOGS \
    --log-interval 5 \
    --warmup-updates $((6*$MAX_UPDATE/100)) \
    --max-epoch $MAX_EPOCH \
    --valid-subset $VALID_SUBSET
