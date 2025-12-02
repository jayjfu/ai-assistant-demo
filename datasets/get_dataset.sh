DEMONSTRATION_DATA_DIR="./demonstration_data"
COMPARISON_DATA_DIR="./comparison_data"

mkdir $DEMONSTRATION_DATA_DIR
mkdir $COMPARISON_DATA_DIR

DATASET_NAME="open-assistant-oasst2"
mkdir -p $DEMONSTRATION_DATA_DIR/$DATASET_NAME
cd $DEMONSTRATION_DATA_DIR/$DATASET_NAME

wget https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz
gunzip 2023-11-05_oasst2_ready.trees.jsonl.gz

cd -