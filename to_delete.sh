#DATA_PATH='data-barthez-4splits/data-barthez1.1:data-barthez-4splits/data-barthez1.2:data-barthez-4splits/data-barthez2.1:data-barthez-4splits/data-barthez2.2'
DATA_PATH='pretraining_data/data-bin-fr'
MAX_UPDATE=105000
MODEL_PATH='samll_bart.pt'
SAVE_DIR=checkpoints_small_model_$1
MAX_SENTENCES=64
MAX_TOKENS=6000
MASK=0.3
PERMUTE_SENTENCES=1
SAVE_INTERVAL=5000
LR=0.0004
TENSORBOARD_LOGS=tensorboard_logs_bart_small/$1
SPM_MODEL_PATH=sentence_piece_multilingual.model
MAX_EPOCH=4

fairseq-train $DATA_PATH \
    --optimizer=adam \
    --adam-betas='(0.9, 0.999)' \
    --adam-eps=1e-06 \
    --arch='bart_large' \
    --bpe='sentencepiece' \
    --sentencepiece-vocab=$SPM_MODEL_PATH \
    --clip-norm=0.1 \
    --log-interval=100 \
    --mask=$MASK \
    --mask-length='span-poisson' \
    --mask-random=0.1 \
    --permute-sentences=$PERMUTE_SENTENCES \
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
    --layernorm-embedding \
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
    --distributed-world-size=1 \
    --distributed-port 12345 \
    --dropout 0.1 \
    --dataset-impl 'mmap' \
    --max-epoch $MAX_EPOCH \
    --warmup-updates $((6*$MAX_UPDATE/100)) \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-layers 2 \
    --encoder-attention-heads 8 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-layers 2 \
    --decoder-attention-heads 8 \
    --no-epoch-checkpoints \
    --valid-subset valid,valid_multilingual

