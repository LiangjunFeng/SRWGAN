from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
from utils import util
from classifiers import classifier
from classifiers import classifier2
from classifiers import classifier3
import sys
from models import model
import numpy as np
import time
from losses.center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin
from modules import *


def run(opt):
    def GetNowTime():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    print(GetNowTime())
    since = time.time()

    sys.stdout.flush()

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # load data
    data = util.load_data(opt)
    print("==================================================================================")
    if opt.generalized:
        print("          FLO: generalized "+str(opt.num_shots)+"-shot learning task is beginning!!!")
    else:
        print("          FLO: "+str(opt.num_shots)+"-shot learning task is beginning!!!")
    if opt.num_shots > 0:
        print("   The number of seen classes: " + str(opt.nclass_seen-opt.nclass_unseen) + ", the number of unseen classes: " + str(
            opt.nclass_unseen))
    else:
        print("   The number of seen classes: " + str(opt.nclass_seen) + ", the number of unseen classes: " + str(opt.nclass_unseen))
    print("   The number of training classes: "+str(opt.nclass_seen)+", the number of training samples: "+str(data.ntrain) )
    print("   The number of test unseen samples: "+str(data.test_unseen_feature.size(0)))
    if opt.generalized:
        print("   The number of test seen samples: "+str(data.test_seen_feature.size(0)))
    else:
        print("   The number of test seen samples: 0")
    print("==================================================================================")

    # initialize generator and discriminator
    netG = model.MLP_G(opt)
    mapping = model.Mapping(opt)
    bilinear = model.Bilinear(opt)
    bilinear2 = model.Bilinear(opt)

    # classification loss, Equation (4) of the paper
    cls_criterion = nn.CrossEntropyLoss()
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.MSELoss()

    if opt.dataset in ['CUB', 'FLO']:
        center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
    else:
        raise ValueError('Dataset %s is not supported' % (opt.dataset))

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    input_unseen_att = torch.FloatTensor(opt.batch_size, opt.attSize)

    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    input_label = torch.LongTensor(opt.batch_size)
    input_unseen_label = torch.LongTensor(opt.batch_size)

    if opt.cuda:
        mapping.cuda()
        netG.cuda()
        bilinear.cuda()
        bilinear2.cuda()
        input_res = input_res.cuda()
        noise, input_att = noise.cuda(), input_att.cuda()
        input_unseen_att = input_unseen_att.cuda()
        cls_criterion.cuda()
        input_label = input_label.cuda()
        input_unseen_label = input_unseen_label.cuda()
        criterion1.cuda()
        criterion2.cuda()
        criterion3.cuda()

    def sample():
        batch_feature, batch_label, batch_att, batch_unseen_att, batch_unseen_label = data.next_batch_all_class_and_unseen(opt.batch_size) # no real unseen samples are used!!
        input_res.copy_(batch_feature)
        input_att.copy_(batch_att)
        input_label.copy_(util.map_label(batch_label, data.seenclasses))
        input_unseen_att.copy_(batch_unseen_att)
        input_unseen_label.copy_(util.map_label(batch_unseen_label, data.unseenclasses))

    def generate_syn_feature(netG, classes, attribute, num):
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
        syn_label = torch.LongTensor(nclass * num)
        syn_att = torch.FloatTensor(num, opt.attSize)
        if opt.cuda:
            syn_att = syn_att.cuda()

        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            output, _, _, _ = netG(syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)

        return syn_feature, syn_label

    def map_label(label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i

        return mapped_label

    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    # setup optimizer
    optimizerD = optim.Adam(mapping.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerB = optim.Adam(bilinear.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerB2 = optim.Adam(bilinear2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def calc_gradient_penalty(netD, real_data, fake_data, input_att):
        # print real_data.size()
        alpha = torch.rand(opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if opt.cuda:
            alpha = alpha.cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)
        _, _, disc_interpolates, _, _ = netD(interpolates, input_att)
        ones = torch.ones(disc_interpolates.size())
        if opt.cuda:
            ones = ones.cuda()
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=ones,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
        return gradient_penalty

    beta = 0
    best_gzsl_acc = 0
    best_t = 0
    begin_time = time.time()
    run_time1 = 0
    run_time2 = 0
    for epoch in range(opt.nepoch):
        FP = 0
        mean_lossD = 0
        mean_lossG = 0

        for i in range(0, data.ntrain, opt.batch_size):

            for p in mapping.parameters():
                p.requires_grad = True

            for iter_d in range(opt.critic_iter):
                sample()
                mapping.zero_grad()
                center_criterion.zero_grad()

                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                muR, varR, criticD_real, latent_pred, _ = mapping(input_resv, input_attv)
                criticD_real = criticD_real.mean()

                fake, _, _, _ = netG(input_att)
                muF, varF, criticD_fake, _, _ = mapping(fake.detach(), input_att)
                criticD_fake = criticD_fake.mean()

                gradient_penalty = calc_gradient_penalty(mapping, input_res, fake.data, input_att)
                mi_loss = MI_loss(torch.cat((muR, muF), dim=0), torch.cat((varR, varF), dim=0), opt.i_c)
                center_loss = opt.center_weight * center_criterion(muR, input_label, margin=opt.center_margin)

                D_cost = criticD_fake - criticD_real + gradient_penalty + 0.001 * criticD_real ** 2 + beta * mi_loss + center_loss
                D_cost.backward()

                optimizerD.step()
                beta = optimize_beta(beta, mi_loss.item())
                optimizer_center.step()

            for p in mapping.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation

            netG.zero_grad()
            bilinear.zero_grad()
            bilinear2.zero_grad()

            fake, fake_noise1, fake_noise2, fake_noise3 = netG(input_att)
            fake_unseen, fake_unseen_noise1, fake_unseen_noise2, fake_unseen_noise3 = netG(input_unseen_att)

            seen_z1_mean = get_mean(fake_noise1, input_label, opt.nclass_seen)
            score_SBC1 = bilinear(input_res.cuda(), seen_z1_mean.cuda())
            loss_SBC1 = cls_criterion(score_SBC1, input_label)
            labels = Variable(input_label.view(opt.batch_size, 1))
            real_proto = Variable(data.real_proto.cuda())
            dists1 = pairwise_distances(fake, real_proto)
            min_idx1 = torch.zeros(opt.batch_size, data.train_cls_num)
            for i in range(data.train_cls_num):
                min_idx1[:, i] = torch.min(dists1.data[:, i * opt.n_clusters:(i + 1) * opt.n_clusters], dim=1)[
                                     1] + i * opt.n_clusters
            min_idx1 = Variable(min_idx1.long().cuda())
            bias_eli_loss = opt.proto_param2*dists1.gather(1, min_idx1).gather(1, labels).squeeze().view(-1).mean()
            unseen_z1_mean = get_mean(fake_unseen_noise1, input_unseen_label, opt.nclass_unseen)
            score_UBC1 = bilinear(fake_unseen.cuda(), unseen_z1_mean.cuda())
            loss_UBC1 = entropy(score_UBC1)
            loss_CBC1 = criterion1(torch.mm(seen_z1_mean.T.cuda(), seen_z1_mean.cuda()),
                                   torch.mm(unseen_z1_mean.T.cuda(), unseen_z1_mean.cuda()))
            if epoch > opt.k: bias_eli_loss += opt.structure_weight*(opt.a * loss_SBC1 + opt.b * loss_UBC1 + opt.c * loss_CBC1)
            else: bias_eli_loss += opt.structure_weight*(opt.a * loss_SBC1 + opt.c * loss_CBC1)

            seen_z2_mean = get_mean(fake_noise2, input_label, opt.nclass_seen)
            score_SBC2 = bilinear2(input_res.cuda(), seen_z2_mean.cuda())
            loss_SBC2 = entropy(score_SBC2)
            seen_feature, seen_label = generate_syn_feature_with_grad(netG, data.seenclasses, data.attribute,
                                                                      opt.loss_syn_num, opt)
            seen_mapped_label = map_label(seen_label, data.seenclasses)
            transform_matrix = torch.zeros(data.train_cls_num, seen_feature.size(0))
            for i in range(data.train_cls_num):
                sample_idx = (seen_mapped_label == i).nonzero().squeeze()
                if sample_idx.numel() == 0:
                    continue
                else:
                    cls_fea_num = sample_idx.numel()
                    transform_matrix[i][sample_idx] = 1 / cls_fea_num * torch.ones(1, cls_fea_num).squeeze()
            transform_matrix = Variable(transform_matrix.cuda())
            fake_proto = torch.mm(transform_matrix, seen_feature)
            dists2 = pairwise_distances(fake_proto, Variable(data.real_proto.cuda()))
            min_idx2 = torch.zeros(data.train_cls_num, data.train_cls_num)
            for i in range(data.train_cls_num):
                min_idx2[:, i] = torch.min(dists2.data[:, i * opt.n_clusters:(i + 1) * opt.n_clusters], dim=1)[
                                     1] + i * opt.n_clusters
            min_idx2 = Variable(min_idx2.long().cuda())
            lbl_idx = Variable(torch.LongTensor(list(range(data.train_cls_num))).cuda())
            aux_loss = opt.proto_param1*dists2.gather(1, min_idx2).gather(1, lbl_idx.unsqueeze(1)).squeeze().mean()
            unseen_z2_mean = get_mean(fake_unseen_noise2, input_unseen_label, opt.nclass_unseen)
            loss_CBC2 = criterion2(torch.mm(seen_z2_mean.T.cuda(), seen_z2_mean.cuda()),
                                   torch.mm(unseen_z2_mean.T.cuda(), unseen_z2_mean.cuda()))
            aux_loss += opt.structure_weight * (opt.a * loss_SBC2 + opt.c * loss_CBC2)

            seen_z3_mean = get_mean(fake_noise3, input_label, opt.nclass_seen)
            unseen_z3_mean = get_mean(fake_unseen_noise3, input_unseen_label, opt.nclass_unseen)
            random_loss = opt.c * criterion3(torch.mm(seen_z3_mean.T.cuda(), seen_z3_mean.cuda()),
                                             torch.mm(unseen_z3_mean.T.cuda(), unseen_z3_mean.cuda()))
            random_loss = opt.structure_weight * random_loss

            _, _, criticG_fake, latent_pred_fake, _ = mapping(fake, input_att, train_G=True)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake

            seen_res_mean = get_mean(input_res, input_label, opt.nclass_seen).cuda()
            c_errG_fake = opt.cls_weight*cls_criterion(torch.mm(fake, seen_res_mean.t()), input_label)

            errG = G_cost + c_errG_fake + bias_eli_loss + aux_loss + random_loss

            errG.backward()
            optimizerG.step()
            optimizerB.step()
            optimizerB2.step()

        print('EP[%d/%d]************************************************************************************' % (
            epoch, opt.nepoch))

        netG.eval()
        if opt.generalized:
            if epoch % 10 == 0:
                # Generalized zero-shot learning
                if opt.generalized:
                    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
                    train_X = torch.cat((data.train_feature, syn_feature), 0)
                    train_Y = torch.cat((data.train_label, syn_label), 0)

                    nclass = opt.nclass_all
                    if opt.two_stage:
                        gzsl_cls = classifier2.CLASSIFIER(mapping, train_X, train_Y, data, nclass, opt.cuda, opt,
                                                          opt.classifier_lr, 0.5, 50,
                                                          2 * opt.syn_num, True)
                    else:
                        gzsl_cls = classifier3.CLASSIFIER(mapping, train_X, train_Y, data, nclass, opt.cuda, opt,
                                                          opt.classifier_lr, 0.5, 50,
                                                          2 * opt.syn_num, True)
                    if best_gzsl_acc <= gzsl_cls.H:
                        best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
                        run_time1 = time.time() - begin_time
                    print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f, run_time=%.4f'
                          % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H, best_acc_seen, best_acc_unseen, best_gzsl_acc, run_time1))
        else:
            if epoch % 10 == 0:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
                cls = classifier2.CLASSIFIER(mapping, syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                             data.unseenclasses.size(0), opt.cuda, opt, opt.classifier_lr, 0.5, 50,
                                             2 * opt.syn_num,
                                             False, opt.ratio, epoch)
                if best_t < cls.acc:
                    best_t = cls.acc
                    run_time2 = time.time() - begin_time
                acc = cls.acc
                print('ZSL: unseen class accuracy= %.4f, best_t=%.4f, run_time=%.4f '%(acc, best_t, run_time2))
        netG.train()

    time_elapsed = time.time() - since
    print('End run!!!')
    print('Time Elapsed: {}'.format(time_elapsed))
    print(GetNowTime())
