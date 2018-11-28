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
    print(filename)
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
        if 'Epoch' in line and 'Val   Top 5 All  Acc' in line:
            sp_line = line.split()
            acc = float(sp_line[7])
            if acc > max_val:
                max_val = acc
                max_idx = idx
                max_epoch = int(sp_line[1])
            last_epoch = int(sp_line[1])

    val_acc1 = float(lines[max_idx-2].split()[7])
    val_acc5 = float(lines[max_idx].split()[7])
    tes_acc1 = float(lines[max_idx+2].split()[7])
    tes_acc5 = float(lines[max_idx+4].split()[7])
    #print(comm_str)
    print("Acc 1: %.4f / %.4f     Acc 5: %.4f / %.4f    Epoch: %d / %d" \
           % (val_acc1, tes_acc1, val_acc5, tes_acc5, max_epoch, last_epoch))
    print(lines[max_idx+2])
    #print(lines[max_idx+4])
    print("")


if __name__ == '__main__':
    folder = sys.argv[1]
    EDIM = int(sys.argv[2])

    MAX_THR=0
    DIS_STEP=5
    NUM_CLASSES=1240

    LFUNC='logistic'
    NUM_EPOCHS=400
    BSIZE=128

    NUM_TR_LAYERS_LIST=[3]
    EXP_LIST=[0.6, 0.75, 1.0, 1.3, 1.5]
    KEEP_PROB_LIST=[1.0]
    LEARNING_RATE_LIST=[0.00001, 0.0001]

    for NTR_LAYERS in NUM_TR_LAYERS_LIST:
        for EX in EXP_LIST:
            for KPROB in KEEP_PROB_LIST:
                for LR in LEARNING_RATE_LIST:
                    print('Num layers: '+str(NTR_LAYERS)+'  Exp: '+str(EX)+'  Keep prob: '+str(KPROB)+'  L rate: '+str(LR))
                    OUT='IMGNET_EMB'+str(EDIM)+'_NLAYERS'+str(NTR_LAYERS)+'_LFUNC'+LFUNC+'_KPROB'+str(KPROB)+'_EXP'+str(EX)+'_LR'+str(LR)+'_BS'+str(BSIZE)
                    filename = os.path.join(folder, OUT)
                    parse_results(filename)
            print('')
            print('')
            print('')
            print('')
