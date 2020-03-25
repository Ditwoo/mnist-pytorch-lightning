
DATA_DIR=data/splits

python src/main.py \
--train_data $DATA_DIR/train.csv \
--valid_data $DATA_DIR/valid.csv \
--lr 0.001 \
--gpus 1
# --gpus 2 --distributed_backend dp