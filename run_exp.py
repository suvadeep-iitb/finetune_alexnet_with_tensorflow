#!/usr/bin/env python
import subprocess, time, csv
from multiprocessing import Pool


QUEUE_SIZE = 3
SLEEP_TIME = 1 #in minutes
WAIT_TIME = 4*60 #in minutes


max_trial = WAIT_TIME//SLEEP_TIME
def execute_command(command_tuple):
  qsub_command = command_tuple[0]
  command_id = command_tuple[1]
  tmp_file = 'tmp/comm_'+str(command_id)
  trial = 0
  while(True):
    exit_status = subprocess.call(qsub_command, shell=True, stdout=open(tmp_file, 'w'))
    if exit_status is 1:  # Check to make sure the job submitted
      print("Job %s failed to submit" % qsub_command)
      return
    line = open(tmp_file).readline()
    if '.sdb' in line:
      l = line.split()
      job = l[0]
      print('Job started: '+job)
      break
    else:
      trial += 1
      time.sleep(SLEEP_TIME*60)
    if trial > max_trial:
      print("Failed to execute command: "+qsub_command)
      return

  time.sleep(SLEEP_TIME*60)
  while(True):
    check_command = 'qstat -n '+job
    with open(tmp_file, 'w') as f:
      exit_status = subprocess.call(check_command, shell=True, stdout=f, stderr=f)
      if exit_status is 1:  # Check to make sure the job submitted
        print("Job %s failed to submit" % qsub_command)
        return
    lines = open(tmp_file).readlines()
    line = ' '.join(lines)
    if 'Job has finished' in line:
        print('Job completed: '+job)
        break
    time.sleep(SLEEP_TIME*60)

  subprocess.call('rm '+tmp_file, shell=True)
    

command_list = []
with open('params_file.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  for count, param_dict in enumerate(reader):
    TRPATHS = param_dict['TRAIN_PATHS']
    VPATHS = param_dict['VAL_PATHS']
    TEPATHS = param_dict['TEST_PATHS']
    CPATH = param_dict['CHP_PATH']
    FPATH = param_dict['FW_PATH']

    MTHR = param_dict['MAX_THR']
    DSTEP = param_dict['DIS_STEP']
    NCLASSES = param_dict['NUM_CLASSES']
    EDIM = param_dict['EMB_DIM']

    NTR_LAYERS = param_dict['NUM_TR_LAYERS']
    LFUNC = param_dict['LOSS_FUNC']
    KPROB = param_dict['KEEP_PROB']
    EX = param_dict['EXP']
    LR = param_dict['LEARNING_RATE']
    NEPOCHS = param_dict['NUM_EPOCHS']
    BSIZE = param_dict['BATCH_SIZE']

    
    OUT='IMGNET_EMB'+str(EDIM)+'_NLAYERS'+str(NTR_LAYERS)+'_LFUNC'+LFUNC+'_KPROB'+str(KPROB)+'_EXP'+str(EX)+'_LR'+str(LR)+'_BS'+str(BSIZE)
    qsub_command = 'qsub -v TRAIN_PATHS='+TRPATHS+ \
                   ',VAL_PATHS='+VPATHS+ \
                   ',TEST_PATHS='+TEPATHS+ \
                   ',CHP_PATH='+CPATH+ \
                   ',FW_PATH='+FPATH+ \
                   ',MAX_THR='+str(MTHR)+ \
                   ',DIS_STEP='+str(DSTEP)+ \
                   ',NUM_CLASSES='+str(NCLASSES)+ \
                   ',EMB_DIM='+str(EDIM)+ \
                   ',NUM_TR_LAYERS='+str(NTR_LAYERS)+ \
                   ',LOSS_FUNC='+LFUNC+ \
                   ',KEEP_PROB='+str(KPROB)+ \
                   ',EXP='+str(EX)+ \
                   ',LEARNING_RATE='+str(LR)+ \
                   ',NUM_EPOCHS='+str(NEPOCHS)+ \
                   ',BATCH_SIZE='+str(BSIZE)+ \
                   ',OUTPUT='+OUT+ \
                   ' gpu_exp.pbs'
    command_list.append((qsub_command, count))

command_exe_pool = Pool(QUEUE_SIZE)
command_exe_pool.map(execute_command, command_list)

