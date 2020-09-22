DATA_SET='XNLI'
MODEL='M_fr_multilingual'
TASK='sentence_prediction'
DATA_PATH='../../../FLUE_data/data-FLUE-XNLI'
MODEL_PATH='../../../checkpoints/denoising/multilingual_fr/ms64_mu75000_si5000_lr0.0004_me20_dws4/checkpoint_last.pt'
MAX_SENTENCES=32
MAX_UPDATE=81800
LR=5e-05
MAX_EPOCH=40
DISTRIBUTED_WORLD_SIZE=1
SENTENCE_PIECE_MODEL='../../../sentence_piece_multilingual.model'
VALID_SUBSET='valid,test'
NUM_CLASSES=3

TENSORBOARD_LOGS=../../../tensorboard_logs/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}
SAVE_DIR=../../../checkpoints/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}
LOGS_DIR=../../../logs/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}

CUDA_VISIBLE_DEVICES=0
mkdir -p $LOGS_DIR

fairseq-train $DATA_PATH \
    --restore-file $MODEL_PATH \
    --max-sentences $MAX_SENTENCES \
    --task $TASK \
    --update-freq 6 \
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
    --valid-subset $VALID_SUBSET 2>&1 | tee $LOGS_DIR/train.log
