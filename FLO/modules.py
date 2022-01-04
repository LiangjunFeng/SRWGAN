import torch
import torch.nn as nn
import math


def generate_syn_feature_with_grad(netG, classes, attribute, num, opt):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature, _, _, _ = netG(syn_att)
    return syn_feature, syn_label.cpu()

def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2) - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
    MI_loss = (torch.mean(kl_divergence) - i_c)
    return MI_loss


def optimize_beta(beta, MI_loss, alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))
    return beta_new


def get_mean(data, label, nclass):
    data_mean = torch.FloatTensor(nclass, data.size(1))
    for i in range(nclass):
        idx_iclass = label.eq(i)
        data_mean[i] = data[idx_iclass].mean(dim=0)
    return data_mean


def compute_rank_correlation(att_map, att_gd):
    n = torch.tensor(att_map.shape[1])
    upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
    down = n * (n.pow(2) - 1.0)
    return (1.0 - (upper / down))


class DotProductSimilarity(nn.Module):
    def __init__(self, scale_output=True):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if (self.scale_output):
            result /= math.sqrt(tensor_1.size(-1))
        return result


def get_similarity(data, metric):
    n = data.size(0)
    input1 = data.repeat_interleave(n, dim=0).float().cuda()
    input2 = data.repeat(n, 1).float().cuda()
    if metric == 'cosine':
        similarity = torch.cosine_similarity(input1, input2, dim=1)
    elif metric == 'euclidean':
        similarity = 1 / (1 + torch.norm(input1 - input2, dim=1, p=2))
    elif metric == 'cityblock':
        similarity = 1 / (1 + torch.norm(input1 - input2, dim=1, p=1))
    elif metric == 'rank':
        similarity = compute_rank_correlation(input1.sort(dim=1)[1].float().cuda(),
                                              input2.sort(dim=1)[1].float().cuda())
    elif metric == 'dot':
        sim = DotProductSimilarity()
        similarity = sim(input1.sort(dim=1)[1].float().cuda(), input2.sort(dim=1)[1].float().cuda())
    return similarity.view(n, n).float().cuda()


def entropy(energy):
    energy = torch.softmax(energy, dim=1)
    log_energy = torch.log(energy + 1e-8)
    out = -1 * energy.mul(log_energy)
    return out.sum(dim=-1).mean()


def en(proba, classes):
    a = torch.ones(proba.shape[1])
    for i in classes:
        a[i] = 0
    proba = proba.cuda().double() * a.cuda().double()
    return proba.max(1)[1]