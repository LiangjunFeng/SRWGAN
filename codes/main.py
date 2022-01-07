import argparse
from srwgan import run

#=================================================================================================================================================================
#                                                           The Codes of SRWGAN for FLO on ZSL, GZSL, FSL, and GFSL
# For each setting, we give the COMMAND LINE for execution and RESULTS for reference.
# For ZSL and GZSL, We use ResNet101 and Char-CNN-RNN embeddings. The splits contributed by Xian etal. are used.
# For FSL and GFSL, a few unseen samples are added into the training set, which is performed based on Xian's ZSL splits by ourselves.
# Pytorch=1.2.0 and one 2080 Ti GPU are used for model training.
#=================================================================================================================================================================
# Note: To see the log in Ubuntu system, use the "cat" command, like "cat flo0.log"   
#
# GZSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset FLO --a 1e-2 --b 1e-3 --c 1e-3 --nz 400 --center_weight 0.001 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.005  --structure_weight 1e-1  > flo0.log 2>&1 &
# best_seen=0.956, best_unseen=0.718, best_h=0.820, run_time=1678.6051
#
# ZSL setting
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized False --dataset FLO --a 1e-4 --b 1e-3 --c 1e-3 --nz 200 --center_weight 0.01 --syn_num 1600 --batch_size 2048 --i_c 0.1 --cls_weight 0.005  --structure_weight 1e-1  > flo1.log 2>&1 &
# best_t=0.7808, run_time=1338.8009
#
# GFSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset FLO --num_shots 1 --a 1e-2 --b 1e-3 --c 1e-3 --nz 400 --center_weight 0.001 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 1  --structure_weight 1e-1 --nepoch 250 > flo0.log 2>&1 &
# 1-shot GFSL: best_seen=0.964, best_unseen=0.740, best_h=0.837, run_time=2612.5557
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset FLO --num_shots 3 --a 1e-2 --b 1e-3 --c 1e-3 --nz 400 --center_weight 0.001 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 250 > flo1.log 2>&1 &
# 3-shot GFSL: best_seen=0.943, best_unseen=0.832, best_h=0.884, run_time=1391.6268
# CUDA_VISIBLE_DEVICES=2 python3 -u main.py --dataset FLO --num_shots 5 --a 1e-2 --b 1e-3 --c 1e-3 --nz 400 --center_weight 0.001 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 250 > flo2.log 2>&1 &
# 5-shot GFSL: best_seen=0.957, best_unseen=0.854, best_h=0.903, run_time=1548.3417
# CUDA_VISIBLE_DEVICES=3 python3 -u main.py --dataset FLO --num_shots 10 --a 1e-2 --b 1e-3 --c 1e-3 --nz 400 --center_weight 0.001 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 250 > flo3.log 2>&1 &
# 10-shot GFSL: best_seen=0.964, best_unseen=0.899, best_h=0.931, run_time=1595.5786
#
# FSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --generalized False  --dataset FLO --num_shots 1 --a 1e-4 --b 1e-3 --c 1e-3 --nz 200 --center_weight 0.01 --syn_num 1600 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 1  --structure_weight 1e-1 --nepoch 500  > flo0.log 2>&1 &
# 1-shot FSL: best_t=0.7844, run_time=4918.3342
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --generalized False  --dataset FLO --num_shots 3 --a 1e-4 --b 1e-3 --c 1e-3 --nz 200 --center_weight 0.01 --syn_num 1600 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 500 > flo1.log 2>&1 &
# 3-shot FSL:  best_t=0.9035, run_time=2853.1971
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized False  --dataset FLO --num_shots 5 --a 1e-4 --b 1e-3 --c 1e-3 --nz 200 --center_weight 0.01 --syn_num 1600 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 500 > flo2.log 2>&1 &
# 5-shot FSL: best_t=0.9248, run_time=3054.4516
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized False  --dataset FLO --num_shots 10 --a 1e-4 --b 1e-3 --c 1e-3 --nz 200 --center_weight 0.01 --syn_num 1600 --batch_size 2048 --i_c 0.1 --cls_weight 0.005 --n 3 --structure_weight 1e-1 --nepoch 500 > flo3.log 2>&1 &
# 10-shot FSL: best_t=0.9239, run_time=2768.2657
#=================================================================================================================================================================

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--generalized', default=True, type = str2bool, help='enable generalized zero-shot learning')
parser.add_argument('--num_shots', type=int, default=0, help='the number of shots')
parser.add_argument('--i_c', type=float, default=0.3, help='information constrain')
parser.add_argument('--center_margin', type=float, default=190, help='the margin in the center loss')
parser.add_argument('--center_weight', type=float, default=0.1, help='the weight for the center loss')
parser.add_argument('--structure_weight', type=float, default=1, help='the weight for the sr loss')
parser.add_argument('--syn_num', type=int, default=600, help='number features to generate per class')
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--a', type=float, default=1e-8, help='the weight for loss1')
parser.add_argument('--b', type=float, default=1e-7, help='the weight for loss2')
parser.add_argument('--c', type=float, default=1e-9, help='the weight for loss3')
parser.add_argument('--nz', type=int, default=85, help='size of the semantic representation * 2')
parser.add_argument('--nepoch', type=int, default=500, help='the number of epochs')
parser.add_argument('--n', type=int, default=3, help='the centers')
args = parser.parse_args()

class myArgs():
    def __init__(self, args):
        self.dataset = args.dataset; self.generalized = args.generalized; self.i_c = args.i_c; self.center_margin = args.center_margin
        self.center_weight = args.center_weight; self.n = args.n; self.structure_weight = args.structure_weight
        self.syn_num = args.syn_num; self.batch_size = args.batch_size; self.nepoch = args.nepoch; self.cls_weight = args.cls_weight
        self.nz = args.nz; self.a = args.a; self.b = args.b; self.c = args.c; self.num_shots = args.num_shots; self.two_stage = True; self.workers = 2
        self.latenSize = 1024; self.beta1 = 0.5; self.ngpu = 1; self.start_epoch = 0; self.ratio = 0.2; self.loss_syn_num = 20
        self.metric = 'cosine';self.param1 = 1e-1; self.param2 = 3e-2; self.ratio = 0.4; self.manualSeed = 806; self.preprocessing = True
        self.cuda = True; self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001; self.classifier_lr = 0.001; self.lambda1 = 10; self.critic_iter = 5
        self.nclass_all = 102; self.attSize = 1024; self.resSize = 2048; self.k = 5; self.nclass_unseen = 20
        if self.num_shots == 0:
            self.nclass_seen = 82
        else:
            self.nclass_seen = self.nclass_all

opt = myArgs(args)
run(opt)













