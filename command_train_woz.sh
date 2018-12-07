
MODEL=gce_woz
DATA=woz
GPU=0
EPOCH=200

CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU -n $MODEL --data $DATA --epoch $EPOCH
