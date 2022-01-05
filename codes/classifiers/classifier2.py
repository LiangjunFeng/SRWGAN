import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
import numpy as np
from scipy.stats import entropy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class CLASSIFIER:
    def __init__(self, mapping, _train_X, _train_Y, data_loader, _nclass, _cuda, opt, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True,ratio=0.6,epoch=20):
        self.train_X, _, _, _, _ = mapping(_train_X.cuda(), torch.rand(_train_X.size(0), opt.attSize).cuda())
        self.train_Y = _train_Y

        self.test_seen_feature, _, _, _, _ = mapping(data_loader.test_seen_feature.cuda(), torch.rand(data_loader.test_seen_feature.size(0), opt.attSize).cuda())
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature, _, _, _, _ = mapping(data_loader.test_unseen_feature.cuda(), torch.rand(data_loader.test_unseen_feature.size(0), opt.attSize).cuda())
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.test_seenclasses = data_loader.test_seenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX(opt.latenSize, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()

        self.data = data_loader

        self.input = torch.FloatTensor(_batch_size, opt.latenSize)
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.ratio = ratio
        self.epoch = epoch

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.backup_X, _, _, _, _ = mapping(_train_X.cuda(), torch.rand(_train_X.size(0), opt.attSize).cuda())
        self.backup_Y = _train_Y

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def pairwise_distances(self,x, y=None):
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
        if y is None:
            dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def fit_zsl(self):
        first_acc=0
        first_all_pred = None
        first_all_output = None

        all_length = self.test_unseen_feature.size(0)
        mapped_test_label = map_label(self.test_unseen_label, self.unseenclasses)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc, pred, output, all_acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if acc > first_acc:
                first_acc = acc
                first_all_pred = pred
                first_all_output = output


        easy_len = int(all_length*self.ratio)
        entropy_value = torch.from_numpy(np.asarray(list(map(entropy, first_all_output.data.cpu()))))
        _, indices = torch.sort(-entropy_value)
        exit_indices = indices[:easy_len]
        keep_indices = indices[easy_len:]

        first_easy_pred = first_all_pred[exit_indices]
        first_easy_label = mapped_test_label[exit_indices]
        first_hard_label = mapped_test_label[keep_indices]
        all_easy_hard_label = torch.cat( (first_easy_label, first_hard_label),0 )

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.backup_X.size()[0] + easy_len
        self.train_X = torch.cat( (self.backup_X, self.test_unseen_feature[exit_indices] ),0 )
        self.train_Y = torch.cat( (self.backup_Y, first_easy_pred ),0 )

        sims = self.pairwise_distances(self.test_unseen_feature[keep_indices], self.train_X)
        value,idx = torch.min(sims,dim=1)
        knn_hard_pred = self.train_Y[idx]
        knn_all_pred = torch.cat( (first_easy_pred,knn_hard_pred),0 )

        acc = self.compute_per_class_acc(all_easy_hard_label,knn_all_pred,self.unseenclasses.size(0))
        if acc > first_acc:
            first_acc = acc
        return first_acc


    def split_pred(self,all_pred, real_label):
        seen_pred = None
        seen_label = None
        unseen_pred = None
        unseen_label = None
        for i in self.test_seenclasses:
            idx = (real_label == i)
            if seen_pred is None:
                seen_pred = all_pred[idx]
                seen_label = real_label[idx]
            else:
                seen_pred = torch.cat( (seen_pred,all_pred[idx]),0 )
                seen_label = torch.cat( (seen_label, real_label[idx]) )

        for i in self.unseenclasses:
            idx = (real_label == i)
            if unseen_pred is None:
                unseen_pred = all_pred[idx]
                unseen_label = real_label[idx]
            else:
                unseen_pred = torch.cat( (unseen_pred,all_pred[idx]),0 )
                unseen_label = torch.cat(  (unseen_label, real_label[idx]), 0 )

        return seen_pred, seen_label, unseen_pred, unseen_label

    # for gzsl
    def fit(self):
        test_seen_length = self.test_seen_feature.shape[0]
        test_unseen_length = self.test_unseen_feature.shape[0]
        all_length = test_seen_length + test_unseen_length
        all_test_feature = torch.cat((self.test_seen_feature, self.test_unseen_feature), 0 )
        all_test_label = torch.cat( (self.test_seen_label, self.test_unseen_label), 0 )
        all_classes = torch.sort(torch.cat( (self.test_seenclasses,self.unseenclasses),0 ))[0]
        first_all_pred = None
        first_all_output = None

        best_H = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen,pred_seen,output_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.test_seenclasses)
            acc_unseen,pred_unseen,output_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

            if H > best_H:
                best_H = H
                first_all_pred = torch.cat( (pred_seen,pred_unseen), 0 )
                first_all_output = torch.cat( (output_seen, output_unseen), 0 )
            if first_all_pred is None:
                first_all_pred = torch.cat((pred_seen, pred_unseen), 0)
                first_all_output = torch.cat((output_seen, output_unseen), 0)
        first_seen_pred,first_seen_label,first_unseen_pred,first_unseen_label = self.split_pred(first_all_pred,all_test_label)
        acc_first_seen = self.compute_per_class_acc_gzsl(first_seen_label, first_seen_pred, self.test_seenclasses)
        acc_first_unseen = self.compute_per_class_acc_gzsl(first_unseen_label, first_unseen_pred,self.unseenclasses)
        acc_first_H = 2*acc_first_seen*acc_first_unseen/(acc_first_seen+acc_first_unseen)

        easy_length = int(all_length*self.ratio)
        entropy_value = torch.from_numpy(np.asarray(list(map(entropy, first_all_output.data.cpu()))))
        _, indices = torch.sort(-entropy_value)
        exit_indices = indices[:easy_length]
        keep_indices = indices[easy_length:]
        first_easy_pred = first_all_pred[exit_indices]
        first_easy_label = all_test_label[exit_indices]
        first_hard_label = all_test_label[keep_indices]
        all_easy_hard_label = torch.cat( (first_easy_label,first_hard_label),0 )

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.backup_X.size(0) + easy_length
        self.train_X = torch.cat( (self.backup_X.cuda(), all_test_feature[exit_indices].cuda()),0 )
        self.train_Y = torch.cat( (self.backup_Y, first_easy_pred),0)

        sims = self.pairwise_distances(all_test_feature[keep_indices], self.train_X)
        value, idx = torch.min(sims, dim=1)
        knn_hard_pred = self.train_Y[idx]
        knn_all_pred = torch.cat( (first_easy_pred,knn_hard_pred),0 )
        knn_seen_pred,knn_seen_label,knn_unseen_pred,knn_unseen_label = self.split_pred(knn_all_pred,all_easy_hard_label)
        acc_knn_seen = self.compute_per_class_acc_gzsl(knn_seen_label,knn_seen_pred,self.test_seenclasses)
        acc_knn_unseen = self.compute_per_class_acc_gzsl(knn_unseen_label,knn_unseen_pred,self.unseenclasses)
        acc_knn_H = 2*acc_knn_seen*acc_knn_unseen/(acc_knn_seen+acc_knn_unseen)

        if acc_first_H < acc_knn_H:
            acc_first_seen = acc_knn_seen
            acc_first_unseen = acc_knn_unseen
            acc_first_H = acc_knn_H

        best_fc_hard_acc = 0
        fc_hard_pred = None
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc,pred,_ = self.val_gzsl(all_test_feature[keep_indices],first_hard_label,all_classes)
            if acc > best_fc_hard_acc:
                fc_hard_pred = pred
        if fc_hard_pred is None:
            fc_hard_pred = pred
        fc_all_pred = torch.cat((first_easy_pred, fc_hard_pred), 0)
        fc_seen_pred, fc_seen_label, fc_unseen_pred, fc_unseen_label = self.split_pred(fc_all_pred, all_easy_hard_label)
        acc_fc_seen = self.compute_per_class_acc_gzsl(fc_seen_label, fc_seen_pred, self.test_seenclasses)
        acc_fc_unseen = self.compute_per_class_acc_gzsl(fc_unseen_label, fc_unseen_pred, self.unseenclasses)
        acc_fc_H = 2 * acc_fc_seen * acc_fc_unseen / (acc_fc_seen + acc_fc_unseen)
        if acc_first_H < acc_fc_H:
            acc_first_seen = acc_fc_seen
            acc_first_unseen = acc_fc_unseen
            acc_first_H = acc_fc_H
        return acc_first_seen, acc_first_unseen, acc_first_H

    def val(self, test_X, test_label, target_classes,second=False):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end].cuda()))
            else:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end]))
            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat( (all_output, output), 0 )
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        acc_all = self.compute_every_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc, predicted_label, all_output, acc_all

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end].cuda()))
            else:
                with torch.no_grad():
                    output = self.model(Variable(test_X[start:end]))

            if all_output is None:
                all_output = output
            else:
                all_output = torch.cat( (all_output, output), 0 )
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc, predicted_label, all_output

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class.mean()

    def compute_every_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx) != 0:
                acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
        return acc_per_class

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
