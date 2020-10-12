DATA_SET='summarization_heading_fr'
MODEL='M_multilingual'
TASK='translation'
DATA_PATH='../../../summarization_fr_data/summarization_heading/data-bin'
MODEL_PATH='../../../checkpoints/denoising/mutilingual/ms64_mu150000_si5000_lr0.0004_me20_dws4/checkpoint_last.pt'
MAX_SENTENCES=32
MAX_SENTENCE_VALID=150
MAX_UPDATE=23952
LR=1e-04
MAX_EPOCH=25
DISTRIBUTED_WORLD_SIZE=1
SENTENCE_PIECE_MODEL='../../../sentence_piece_multilingual.model'
VALID_SUBSET='valid'
SEED=$1
SRC=article
TGT=heading

TENSORBOARD_LOGS=../../../tensorboard_logs/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}/$SEED
SAVE_DIR=../../../checkpoints/$TASK/$DATA_SET/$MODEL/ms${MAX_SENTENCES}_mu${MAX_UPDATE}_lr${LR}_me${MAX_EPOCH}_dws${DISTRIBUTED_WORLD_SIZE}/$SEED

CUDA_VISIBLE_DEVICES=0

fairseq-train $DATA_PATH \
    --restore-file $MODEL_PATH \
    --max-sentences $MAX_SENTENCES \
    --max-sentences-valid $MAX_SENTENCE_VALID \
    --task $TASK \
    --source-lang $SRC --target-lang $TGT \
    --update-freq 1 \
    --seed $SEED \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --eval-scorer eval-precision-recall \
    --best-checkpoint-metric f-1 --maximize-best-checkpoint-metric \
    --eval-scorer-args '{"beam": 5, "max_len_b": 200}' \
    --arch bart_small \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --find-unused-parameters \
    --bpe 'sentencepiece' \
    --sentencepiece-vocab $SENTENCE_PIECE_MODEL \
    --maximize-best-checkpoint-metric \
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
    --valid-subset $VALID_SUBSET \
