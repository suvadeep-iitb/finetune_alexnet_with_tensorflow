#!/usr/bin/env python
import csv

TRAIN_PATHS='500K_imagenet/train_d0.pkl'
VAL_PATHS='500K_imagenet/val_d0.pkl'
TEST_PATHS='500K_imagenet/test_d0.pkl'
CORR_PATH='500K_imagenet/d52_corr.npy'
CHP_PATH='tmp/imagenet/checkpoint/'
FW_PATH='tmp/imagenet/filewriter/'

MAX_THR=0
DIS_STEP=5
NUM_CLASSES=1000

LOSS_FUNC='logistic'
NUM_EPOCHS=400
BATCH_SIZE=128

NUM_TR_LAYERS_LIST=[3]
EMB_DIM_LIST=[50, 10]
EXP_LIST=[0.75, 1.0, 1.3, 1.5]
WGT_LIST=[0.001, 0.1]
KEEP_PROB_LIST=[1.0]
LEARNING_RATE_LIST=[0.000001, 0.00001, 0.0001]


fieldnames = ["TRAIN_PATHS", "VAL_PATHS", "TEST_PATHS", "CORR_PATH", "CHP_PATH", \
              "FW_PATH", "MAX_THR", "DIS_STEP", "NUM_CLASSES", \
              "LOSS_FUNC", "NUM_EPOCHS", "BATCH_SIZE", "NUM_TR_LAYERS", \
              "EMB_DIM", "EXP", "WEIGHT", "KEEP_PROB", "LEARNING_RATE"]

with open('params_file.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  writer.writeheader()
  for NUM_TR_LAYERS in NUM_TR_LAYERS_LIST:
    for EMB_DIM in EMB_DIM_LIST:
      for EXP in EXP_LIST:
        for WEIGHT in WGT_LIST:
          for KEEP_PROB in KEEP_PROB_LIST:
             for LEARNING_RATE in LEARNING_RATE_LIST:
                param_dict = {"TRAIN_PATHS"   : TRAIN_PATHS, \
                              "VAL_PATHS"     : VAL_PATHS, \
                              "TEST_PATHS"    : TEST_PATHS, \
                              "CORR_PATH"     : CORR_PATH, \
                              "CHP_PATH"      : CHP_PATH, \
                              "FW_PATH"       : FW_PATH, \
                              "MAX_THR"       : MAX_THR, \
                              "DIS_STEP"      : DIS_STEP, \
                              "NUM_CLASSES"   : NUM_CLASSES, \
                              "LOSS_FUNC"     : LOSS_FUNC, \
                              "NUM_EPOCHS"    : NUM_EPOCHS, \
                              "BATCH_SIZE"    : BATCH_SIZE, \
                              "NUM_TR_LAYERS" : NUM_TR_LAYERS, \
                              "EMB_DIM"       : EMB_DIM, \
                              "EXP"           : EXP, \
                              "WEIGHT"        : WEIGHT, \
                              "KEEP_PROB"     : KEEP_PROB, \
                              "LEARNING_RATE" : LEARNING_RATE}
                writer.writerow(param_dict)

