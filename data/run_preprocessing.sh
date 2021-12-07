
DATA_DIR="./processed_uncased_blanklines"

TRAIN_INPUT="${DATA_DIR}/oscar_all_uncased.txt"
TRAIN_DATA="${DATA_DIR}/processed_oscar_all_uncased.txt"
DEV_INPUT="${DATA_DIR}/dev.txt"
DEV_DATA="${DATA_DIR}/dev_preprocessed.txt"
TEST_INPUT="${DATA_DIR}/test_input.txt"
TEST_DATA="${DATA_DIR}/test_input_preprocessed.txt"


# Preprocessing
python preprocessing.py \
    --input ${TRAIN_INPUT} \
    --output ${TRAIN_DATA} \
    --save_noised_output

# python preprocessing.py \
#     --input ${DEV_INPUT} \
#     --output ${DEV_DATA}

# python preprocessing.py \
#     --input ${TEST_INPUT} \
#     --output ${TEST_DATA}
