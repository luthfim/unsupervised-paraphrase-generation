# GPU_ID=$1
# TAG=$2
# DEVICE=$3
GPU_ID=0
TAG=test
DEVICE=cpu
LENGTH=32
BATCH_SIZE=64  # batch size per each GPU
LR="6.25e-5"
ACCUM=4
N_EPOCHS=1
SAVE_STEPS=300  # Validating and save checkpoints
RANDOM_SEED=1223734

TRAIN_DATA="./data/id_data/train_preprocessed.txt"
DEV_DATA="./data/id_data/dev_preprocessed.txt"

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --train_data_path ${TRAIN_DATA}\
    --dev_data_path ${DEV_DATA} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${ACCUM} \
    --num_epochs ${N_EPOCHS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LR} \
    --tag ${TAG} \
    --device ${DEVICE} \
    --model cahya/gpt2-small-indonesian-522M \
    --seed ${RANDOM_SEED} #--debug --toy