
MODEL=gce_multiwoz
DATA=multi_woz
GPU=0
EPOCH=200

CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU -n $MODEL --data $DATA --epoch $EPOCH
