import sys, os
import decimal


def f2s(f):
    ctx = decimal.Context()
    ctx.prec = 20
    d1 = ctx.create_decimal(repr(f))
    st = format(d1, 'f')
    if '.' not in st:
        st += '.0'
    return st


def parse_results(filename):
    #print(filename)
    lines = open(filename).readlines()
    lines = [l.strip() for l in lines if l.strip() != ""]

    comm_str = lines[0]
    lines = lines[1:]

    max_val = -1
    max_idx = -1
    max_epoch = 0
    last_epoch = 0
    for idx in range(len(lines)):
        line = lines[idx]
        if 'Epoch' in line and 'Val   Top 5 Acc' in line:
            sp_line = line.split()
            acc = float(sp_line[6])
            if acc > max_val:
                max_val = acc
                max_idx = idx
                max_epoch = int(sp_line[1])
            last_epoch = int(sp_line[1])

    val_acc1 = float(lines[max_idx-1].split()[6])
    val_acc5 = float(lines[max_idx].split()[6])
    tes_acc1 = float(lines[max_idx+1].split()[6])
    tes_acc5 = float(lines[max_idx+2].split()[6])
    #print(comm_str)
    #print("Acc 1: %.4f / %.4f     Acc 5: %.4f / %.4f    Epoch: %d / %d" \
    #       % (val_acc1, tes_acc1, val_acc5, tes_acc5, max_epoch, last_epoch))
    #print(lines[max_idx+2])
    #print("")
    return (val_acc1, val_acc5, tes_acc1, tes_acc5, max_epoch)


if __name__ == '__main__':
    folder = sys.argv[1]
    EDIM = int(sys.argv[2])

    MAX_THR=0
    DIS_STEP=5
    NUM_CLASSES=1000

    LFUNC='boot_soft'
    NUM_EPOCHS=400
    BSIZE=128

    NUM_TR_LAYERS_LIST=[3]
    EXP_LIST=[1.0]
    BETA_LIST=[0.2, 0.35, 0.5, 0.6, 0.75, 1.0]
    C_LIST = [0.01]
    NEL_LIST = [1]
    WGT_LIST=[0]
    KEEP_PROB_LIST=[0.75, 1.0]
    LEARNING_RATE_LIST=[0.00001, 0.0001, 0.001]

    for NTR_LAYERS in NUM_TR_LAYERS_LIST:
      for NEL in NEL_LIST:
        print('Num layers: '+str(NTR_LAYERS)+'  Nel: '+str(NEL))
        for EX in EXP_LIST:
          for C in C_LIST:
            for WGT in WGT_LIST:
              max_val_acc1 = -1
              max_val_acc5 = -1 
              max_tes_acc1 = -1 
              max_tes_acc5 = -1
              max_kp = -1
              max_lr = -1
              max_be = -1
              for BETA in BETA_LIST:
                for KPROB in KEEP_PROB_LIST:
                  #print('KP: '+str(KPROB))
                  for LR in LEARNING_RATE_LIST:
                    OUT='DAC_EMB'+str(EDIM)+'_NLAYERS'+str(NTR_LAYERS)+'_LFUNC'+LFUNC+'_KPROB'+str(KPROB)+'_EXP'+str(EX)+'_BETA'+str(BETA)+'_C'+str(C)+'_NEL'+str(NEL)+'_WGT'+str(WGT)+'_LR'+str(LR)+'_BS'+str(BSIZE)
                    filename = os.path.join(folder, OUT)
                    val_acc1, val_acc5, tes_acc1, tes_acc5, me = parse_results(filename)
                    print('BETA: '+str(BETA)+' K PROB: '+str(KPROB)+' LR: '+str(LR)+'\t'+str(val_acc1)+' / '+str(val_acc5)+'\t'+str(tes_acc1)+' / '+str(tes_acc5)+'\t'+str(me))
                    if max_val_acc5 < val_acc5:
                      max_val_acc1 = val_acc1
                      max_val_acc5 = val_acc5
                      max_tes_acc1 = tes_acc1
                      max_tes_acc5 = tes_acc5
                      max_kp = KPROB
                      max_lr = LR
                      max_be = BETA
                  #print('')
                #print('')
              #print('')
            print(str(max_val_acc1)+' / '+str(max_tes_acc1)+'\t'+str(max_val_acc5)+' / '+str(max_tes_acc5)+'\t'+str(max_be)+' / '+str(max_kp)+' / '+str(max_lr))
            #print('')
            #print('')
        print('')
