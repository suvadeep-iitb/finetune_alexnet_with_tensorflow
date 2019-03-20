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

LOSS_FUNC='softmax'
NUM_EPOCHS=400
BATCH_SIZE=128

NUM_TR_LAYERS_LIST=[4]
EMB_DIM_LIST=[50]
EXP_LIST=[0.5]
C_LIST = [0.01]
NEL_LIST = [2, 3]
WGT_LIST=[0]
KEEP_PROB_LIST=[1.0]
LEARNING_RATE_LIST=[0.00001, 0.0001, 0.001]


fieldnames = ["TRAIN_PATHS", "VAL_PATHS", "TEST_PATHS", "CORR_PATH", "CHP_PATH", \
              "FW_PATH", "MAX_THR", "DIS_STEP", "NUM_CLASSES", \
              "LOSS_FUNC", "NUM_EPOCHS", "BATCH_SIZE", "NUM_TR_LAYERS", \
              "EMB_DIM", "EXP", "C", "NEL", "WEIGHT", "KEEP_PROB", "LEARNING_RATE"]

with open('params_file.csv', 'w') as csvfile:
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  writer.writeheader()
  for NUM_TR_LAYERS in NUM_TR_LAYERS_LIST:
    for EMB_DIM in EMB_DIM_LIST:
      for NEL in NEL_LIST:
        for EXP in EXP_LIST:
          for C in C_LIST:
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
                                "C"             : C, \
                                "NEL"           : NEL, \
                                "WEIGHT"        : WEIGHT, \
                                "KEEP_PROB"     : KEEP_PROB, \
                                "LEARNING_RATE" : LEARNING_RATE}
                  writer.writerow(param_dict)

