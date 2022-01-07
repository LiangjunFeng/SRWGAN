import numpy as np
import scipy.io as sio
import torch
from sklearn.cluster import KMeans
from sklearn import preprocessing

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


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class load_data(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.num_shots = opt.num_shots
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(
                self.train_feature[torch.nonzero(self.train_mapped_label == i), :].numpy(), axis=0)
        n = opt.n
        cn_p = torch.zeros(n * self.train_cls_num, self.feature_dim)
        for i in range(self.train_cls_num):
            sample_idx = (self.train_mapped_label == i).nonzero().squeeze()
            if sample_idx.numel() == 0:
                cn_p[n * i: n * (i + 1)] = torch.zeros(n, self.feature_dim)
            else:
                real_sample_cls = self.train_feature[sample_idx, :]
                if len(real_sample_cls.size()) == 1:
                    real_sample_cls = real_sample_cls.view(-1, 1)
                y_pred = KMeans(n, random_state=3).fit_predict(real_sample_cls)
                for j in range(n):
                    cn_p[n * i + j] = torch.from_numpy(
                        real_sample_cls[torch.nonzero(torch.from_numpy(y_pred) == j), :].mean(dim=0).cpu().numpy())
        self.cn_p = cn_p

    def read_matdataset(self, opt):
        data = sio.loadmat("./dataset/DataOfAnyShot.mat")
        if opt.num_shots == 0:
            splits = sio.loadmat("./dataset/SplitsOfZeroSL.mat")
        elif opt.num_shots == 1:
            splits = sio.loadmat("./dataset/SplitsOfOneSL.mat")
        elif opt.num_shots == 3:
            splits = sio.loadmat("./dataset/SplitsOfThreeSL.mat")
        elif opt.num_shots == 5:
            splits = sio.loadmat("./dataset/SplitsOfFiveSL.mat")
        elif opt.num_shots == 10:
            splits = sio.loadmat("./dataset/SplitsOfTenSL.mat")

        self.feature = data['all_features']
        self.label = data['all_labels'].astype(int).squeeze()
        self.trainval_loc = splits['trainval_index'].squeeze()
        self.test_seen_loc = splits['test_seen_index'].squeeze()
        self.test_unseen_loc = splits['test_unseen_index'].squeeze()
        self.attribute = torch.from_numpy(splits['attributes']).float()
        self.preprocess()

    def preprocess(self):
        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(self.feature[self.trainval_loc])
        _test_seen_feature = scaler.transform(self.feature[self.test_seen_loc])
        _test_unseen_feature = scaler.transform(self.feature[self.test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = self.seenclasses.clone()
        self.test_class = self.unseenclasses.clone()
        self.test_seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.test_seenclasses.size(0) + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()
        self.train_cls_num = self.ntrain_class
        self.test_cls_num = self.ntest_class
        self.n_class = self.test_seenclasses.size(0) + self.ntest_class

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att

    def next_batch_unseen(self, batch_size):  # for transductive learning
        if self.ntest_unseen > batch_size:
            idx = torch.randperm(self.ntest_unseen)[0:batch_size]
            batch_feature = self.test_unseen_feature[idx]
        else:
            idx = torch.randperm(self.ntest_unseen)[0:batch_size - self.ntest_unseen]
            batch_feature = self.test_unseen_feature[idx]
            batch_feature = torch.cat([batch_feature, self.test_unseen_feature], dim=0)
        return batch_feature[torch.randperm(batch_size)]

    def next_batch_all_class_and_unseen(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            if i % self.ntrain_class == 0:
                class_lis = self.train_class[torch.randperm(self.ntrain_class)]
            batch_class[i] = class_lis[i % self.ntrain_class]

        batch_class = batch_class[torch.randperm(batch_size)]
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))

        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            try:
                idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
                idx_file = idx_iclass[idx_in_iclass]
            except Exception:
                idx_file = idx_iclass
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]

        batch_unseen_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            if i % self.ntest_class == 0:
                class_lis = self.test_class[torch.randperm(self.ntest_class)]
            batch_unseen_class[i] = class_lis[i % self.ntest_class]

        batch_unseen_class = batch_unseen_class[torch.randperm(batch_size)]
        batch_unseen_att = self.attribute[batch_unseen_class]

        return batch_feature, batch_label, batch_att, batch_unseen_att, batch_unseen_class