import argparse
from srwgan import run

#=================================================================================================================================================================
#                                                           The Codes of SRWGAN for CUB on ZSL, GZSL, FSL, and GFSL
# For each setting, we give the COMMAND LINE for execution and RESULTS for reference.
# For ZSL and GZSL, We use ResNet101 and attribute embeddings. The splits contributed by Xian etal. are used.
# For FSL and GFSL, a few unseen samples are added into the training set, which is performed based on Xian's ZSL splits by ourselves.
# Pytorch=1.2.0 and one 2080 Ti GPU are used for model training.
#=================================================================================================================================================================
# Note: To see the log in Ubuntu system, use the "cat" command as "cat cub0.log"   
#
# GZSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --generalized True --dataset CUB --nz 300 --a 1e-1 --b 1e-3 --c 1e-3 --center_weight 0.01 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.05  --structure_weight 1e-3  > cub0.log 2>&1 &
# GZSL: best_seen=0.7852, best_unseen=0.6310, best_h=0.6996, run_time=8633.0329
#
# ZSL setting
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized False --dataset CUB --nz 200 --a 1e-3 --b 1e-3 --c 1e-2 --center_weight 0.01 --syn_num 1000 --batch_size 2048 --i_c 0.1 --cls_weight 0.05  --structure_weight 1e-3  > cub1.log 2>&1 &
# ZSL:  best_t=0.7479, run_time=8444.3350
#
# GFSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --generalized True --num_shots 1 --dataset CUB --nz 300 --a 1e-1 --b 1e-3 --c 1e-3 --center_weight 0.01 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 1 --structure_weight 1e-3  > cub0.log 2>&1 &
# 1-shot GFSL: best_seen=0.795, best_unseen=0.660, best_h=0.721, run_time=5965.8138
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized True --num_shots 3 --dataset CUB --nz 300 --a 1e-1 --b 1e-3 --c 1e-3 --center_weight 0.01 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub1.log 2>&1 &
# 3-shot GFSL: best_seen=0.763, best_unseen=0.692, best_h=0.726, run_time=5626.7518
# CUDA_VISIBLE_DEVICES=2 python3 -u main.py --generalized True --num_shots 5 --dataset CUB --nz 300 --a 1e-1 --b 1e-3 --c 1e-3 --center_weight 0.01 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub2.log 2>&1 &
# 5-shot GFSL: best_seen=0.770, best_unseen=0.777, best_h=0.773, run_time=6290.0931
# CUDA_VISIBLE_DEVICES=3 python3 -u main.py --generalized True --num_shots 10 --dataset CUB --nz 300 --a 1e-1 --b 1e-3 --c 1e-3 --center_weight 0.01 --syn_num 800 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub3.log 2>&1 &
# 10-shot GFSL: best_seen=0.773, best_unseen=0.783, best_h=0.778, run_time=6789.4561
#
# FSL setting
# CUDA_VISIBLE_DEVICES=0 python3 -u main.py --generalized False --num_shots 1 --dataset CUB --nz 200 --a 1e-3 --b 1e-3 --c 1e-2 --center_weight 0.01 --syn_num 1000 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 1 --structure_weight 1e-3  > cub0.log 2>&1 &
# 1-shot FSL: best_t=0.7678, run_time=18093.0866
# CUDA_VISIBLE_DEVICES=1 python3 -u main.py --generalized False --num_shots 3 --dataset CUB --nz 200 --a 1e-3 --b 1e-3 --c 1e-2 --center_weight 0.01 --syn_num 1000 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub1.log 2>&1 &
# 3-shot FSL: best_t=0.8106, run_time=12774.7240
# CUDA_VISIBLE_DEVICES=2 python3 -u main.py --generalized False --num_shots 5 --dataset CUB --nz 200 --a 1e-3 --b 1e-3 --c 1e-2 --center_weight 0.01 --syn_num 1000 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub2.log 2>&1 &
# 5-shot FSL: best_t=0.8372, run_time=14971.6963
# CUDA_VISIBLE_DEVICES=3 python3 -u main.py --generalized False --num_shots 10 --dataset CUB --nz 200 --a 1e-3 --b 1e-3 --c 1e-2 --center_weight 0.01 --syn_num 1000 --batch_size 2048 --i_c 0.1 --cls_weight 0.05 --n_clusters 3 --structure_weight 1e-3  > cub3.log 2>&1 &
# 10-shot FSL: best_t=0.8622, run_time=14945.2503
#=================================================================================================================================================================

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--generalized', default=True, type = str2bool, help='enable generalized zero-shot learning')
parser.add_argument('--num_shots', type=int, default=0, help='the number of shots')
parser.add_argument('--i_c', type=float, default=0.3, help='information constrain')
parser.add_argument('--center_margin', type=float, default=190, help='the margin in the center loss')
parser.add_argument('--center_weight', type=float, default=0.1, help='the weight for the center loss')
parser.add_argument('--structure_weight', type=float, default=1, help='the weight for the sr loss')
parser.add_argument('--syn_num', type=int, default=600, help='number features to generate per class')
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--a', type=float, default=1e-8, help='the weight for alignment loss1')
parser.add_argument('--b', type=float, default=1e-7, help='the weight for alignment loss2')
parser.add_argument('--c', type=float, default=1e-9, help='the weight for alignment loss3')
parser.add_argument('--nz', type=int, default=85, help='size of the semantic representation*2')
parser.add_argument('--n_clusters', type=int, default=3, help='the number of clusters')
args = parser.parse_args()

class myArgs():
    def __init__(self, args):
        self.dataset = args.dataset; self.generalized = args.generalized; self.num_shots = args.num_shots
        self.i_c = args.i_c; self.center_margin = args.center_margin; self.center_weight = args.center_weight; self.n_clusters = args.n_clusters
        self.structure_weight = args.structure_weight; self.syn_num = args.syn_num; self.batch_size = args.batch_size
        self.cls_weight = args.cls_weight; self.nz = args.nz; self.a = args.a; self.b = args.b; self.c = args.c; self.workers = 2
        self.latenSize = 1024; self.beta1 = 0.5;  self.ngpu = 1; self.loss_syn_num = 20; self.metric = 'cosine'
        self.proto_param1 = 1e-2; self.proto_param2 = 0.001; self.ratio = 0.2; self.manualSeed = 3483
        self.cuda = True; self.two_stage = True; self.nepoch = 2000; self.ngh = 4096; self.ndh = 4096; self.lr = 0.0001
        self.classifier_lr = 0.001; self.lambda1 = 10; self.critic_iter = 5; self.nclass_all = 200; self.attSize = 312
        self.resSize = 2048; self.nclass_seen = 150; self.k=10; self.nclass_unseen = 50;
        if self.num_shots != 0:
            self.nclass_seen = self.nclass_all
opt = myArgs(args)
run(opt)













