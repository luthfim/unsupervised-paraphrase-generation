GPU_ID=$1
TAG=$2
CHECKPOINT_DIR=$3
GPU_ID=0
TAG=test
CHECKPOINT_DIR='/home/luthfi/workspace/unsupervised-paraphrase-generation/checkpoints/cahya/gpt2-small-indonesian-522M_test_2021-12-03_10:44:49'

T="1.0"  # temperature
k=10
p="1.0"
N_GEN=10
SEED="1234"

INPUT_FILE="./data/id_data/test_input.txt"
PREPROCESSED="./data/id_data/test_input_preprocessed.txt"
TARGET="./data/id_data/test_target.txt"
FILENAME="inferenced_${TAG}_top-${k}-p${p//./_}-T${T//./_}_seed${SEED}.txt"
TAG='test'

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --data_path ${PREPROCESSED} \
    --checkpoint ${CHECKPOINT_DIR} \
    --save "./results/${FILENAME}" \
    --decoding "sampling" \
    --k ${k} \
    --p ${p} \
    --temperature ${T} \
    --num_generate ${N_GEN} \
    --seed ${SEED} \
    --tag ${TAG} \
    --device cpu

# CUDA_VISIBLE_DEVICES=$GPU_ID python postprocessing.py \
#     --input ${INPUT_FILE} \
#     --paraphrase "./results/${FILENAME}" \
#     --output "./results/filtered/${FILENAME}" \
#     --model "bert-base-nli-stsb-mean-tokens" \
#     --tag ${TAG}

# CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
#     --generated "./results/filtered/${FILENAME}" \
#     --ground_truth ${TARGET} \
#     --tag ${TAG}