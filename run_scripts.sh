#!/bin/bash

TRAIN_PATHS='30K_imagenet/train_npy.txt'
VAL_PATHS='30K_imagenet/val_npy.txt'
TEST_PATHS='30K_imagenet/test_npy.txt'
CHP_PATH='tmp/imagenet/checkpoint/'
FW_PATH='tmp/imagenet/filewriter/'

MAX_THR=0
DIS_STEP=1
NUM_CLASSES=1000
EMB_DIM=4096

NUM_TR_LAYERS=2
LOSS_FUNC='logistic'
KEEP_PROB=0.5
EXP=1.0
LEARNING_RATE=0.001
NUM_EPOCHS=10
BATCH_SIZE=128



echo 'python main.py -train_paths='$TRAIN_PATHS' -val_paths='$VAL_PATHS' -test_paths='$TEST_PATHS' -checkpoint_path='$CHP_PATH' -filewriter_path='$FW_PATH' -max_threads='$MAX_THR' -display_step='$DIS_STEP' -num_classes='$NUM_CLASSES' -embedding_dim='$EMB_DIM' -num_train_layers='$NUM_TR_LAYERS' -loss_func='$LOSS_FUNC' -keep_prob='$KEEP_PROB' -exp='$EXP' -learning_rate='$LEARNING_RATE' -num_epochs='$NUM_EPOCHS' -batch_size='$BATCH_SIZE
python main.py -train_paths=$TRAIN_PATHS -val_paths=$VAL_PATHS -test_paths=$TEST_PATHS -checkpoint_path=$CHP_PATH -filewriter_path=$FW_PATH -max_threads=$MAX_THR -display_step=$DIS_STEP -num_classes=$NUM_CLASSES -embedding_dim=$EMB_DIM -num_train_layers=$NUM_TR_LAYERS -loss_func=$LOSS_FUNC -keep_prob=$KEEP_PROB -exp=$EXP -learning_rate=$LEARNING_RATE -num_epochs=$NUM_EPOCHS -batch_size=$BATCH_SIZE
