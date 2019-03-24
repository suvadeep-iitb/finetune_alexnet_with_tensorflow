#!/bin/bash

TRAIN_PATHS='500K_imagenet/train_p2_b100.pkl'
VAL_PATHS='500K_imagenet/val_p2_b100.pkl'
TEST_PATHS='500K_imagenet/test_p2_b100.pkl'
CHP_PATH='tmp/imagenet/checkpoint/'
FW_PATH='tmp/imagenet/filewriter/'

MAX_THR=0
DIS_STEP=5
NUM_CLASSES=100

NUM_TR_LAYERS=6
LOSS_FUNC='softmax'



EMB_DIM=800
KEEP_PROB=0.75
EXP=1.0
NEL=1
C=0.01
LEARNING_RATE=0.0001
NUM_EPOCHS=400
BATCH_SIZE=128


echo 'python3 main.py -train_paths='$TRAIN_PATHS' -val_paths='$VAL_PATHS' -test_paths='$TEST_PATHS' -checkpoint_path='$CHP_PATH' -filewriter_path='$FW_PATH' -max_threads='$MAX_THR' -display_step='$DIS_STEP' -num_classes='$NUM_CLASSES' -embedding_dim='$EMB_DIM' -num_train_layers='$NUM_TR_LAYERS' -loss_func='$LOSS_FUNC' -keep_prob='$KEEP_PROB' -exp='$EXP' -learning_rate='$LEARNING_RATE' -num_epochs='$NUM_EPOCHS' -batch_size='$BATCH_SIZE' -nel='$NEL' -c='$C
python3 main.py -train_paths=$TRAIN_PATHS -val_paths=$VAL_PATHS -test_paths=$TEST_PATHS -checkpoint_path=$CHP_PATH -filewriter_path=$FW_PATH -max_threads=$MAX_THR -display_step=$DIS_STEP -num_classes=$NUM_CLASSES -embedding_dim=$EMB_DIM -num_train_layers=$NUM_TR_LAYERS -loss_func=$LOSS_FUNC -keep_prob=$KEEP_PROB -exp=$EXP -learning_rate=$LEARNING_RATE -num_epochs=$NUM_EPOCHS -batch_size=$BATCH_SIZE -nel=$NEL -c=$C


echo ''
echo ''
echo ''
echo ''


EMB_DIM=800
KEEP_PROB=0.75
EXP=0.3
NEL=1
C=0.01
LEARNING_RATE=0.0001
NUM_EPOCHS=400
BATCH_SIZE=128


echo 'python3 main.py -train_paths='$TRAIN_PATHS' -val_paths='$VAL_PATHS' -test_paths='$TEST_PATHS' -checkpoint_path='$CHP_PATH' -filewriter_path='$FW_PATH' -max_threads='$MAX_THR' -display_step='$DIS_STEP' -num_classes='$NUM_CLASSES' -embedding_dim='$EMB_DIM' -num_train_layers='$NUM_TR_LAYERS' -loss_func='$LOSS_FUNC' -keep_prob='$KEEP_PROB' -exp='$EXP' -learning_rate='$LEARNING_RATE' -num_epochs='$NUM_EPOCHS' -batch_size='$BATCH_SIZE' -nel='$NEL' -c='$C
python3 main.py -train_paths=$TRAIN_PATHS -val_paths=$VAL_PATHS -test_paths=$TEST_PATHS -checkpoint_path=$CHP_PATH -filewriter_path=$FW_PATH -max_threads=$MAX_THR -display_step=$DIS_STEP -num_classes=$NUM_CLASSES -embedding_dim=$EMB_DIM -num_train_layers=$NUM_TR_LAYERS -loss_func=$LOSS_FUNC -keep_prob=$KEEP_PROB -exp=$EXP -learning_rate=$LEARNING_RATE -num_epochs=$NUM_EPOCHS -batch_size=$BATCH_SIZE -nel=$NEL -c=$C


echo ''
echo ''
echo ''
echo ''



EMB_DIM=800
KEEP_PROB=0.75
EXP=0.3
NEL=5
C=0.01
LEARNING_RATE=0.0001
NUM_EPOCHS=400
BATCH_SIZE=128


echo 'python3 main.py -train_paths='$TRAIN_PATHS' -val_paths='$VAL_PATHS' -test_paths='$TEST_PATHS' -checkpoint_path='$CHP_PATH' -filewriter_path='$FW_PATH' -max_threads='$MAX_THR' -display_step='$DIS_STEP' -num_classes='$NUM_CLASSES' -embedding_dim='$EMB_DIM' -num_train_layers='$NUM_TR_LAYERS' -loss_func='$LOSS_FUNC' -keep_prob='$KEEP_PROB' -exp='$EXP' -learning_rate='$LEARNING_RATE' -num_epochs='$NUM_EPOCHS' -batch_size='$BATCH_SIZE' -nel='$NEL' -c='$C
python3 main.py -train_paths=$TRAIN_PATHS -val_paths=$VAL_PATHS -test_paths=$TEST_PATHS -checkpoint_path=$CHP_PATH -filewriter_path=$FW_PATH -max_threads=$MAX_THR -display_step=$DIS_STEP -num_classes=$NUM_CLASSES -embedding_dim=$EMB_DIM -num_train_layers=$NUM_TR_LAYERS -loss_func=$LOSS_FUNC -keep_prob=$KEEP_PROB -exp=$EXP -learning_rate=$LEARNING_RATE -num_epochs=$NUM_EPOCHS -batch_size=$BATCH_SIZE -nel=$NEL -c=$C

echo ''
echo ''


