
DIR=exp/multi_woz/gce
MODEL=gce_multiwoz
DATA=multi_woz
GPU=0
CHECKPOINT=$DIR/$MODEL

CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --gpu $GPU --data $DATA --split test --dsave $CHECKPOINT
