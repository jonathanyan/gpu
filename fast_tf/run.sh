python ./train.py  --config \
    DATA.BASEDIR=/home/jyan/tmp/coco \
    TRAIN.NUM_GPUS=1 TRAIN.STEPS_PER_EPOCH=5 \
    TRAIN.WARMUP=1 
