import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from settings import *


def subsequent_mask(size):
    # 生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它最后两维形成一个方阵
    attn_shape = (1, size, size)
    # 然后使用np.ones方法向这个形状中添加1元素，形成上三角阵
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(mask) == 0


# 标签平滑创建了一个分布，该分布设定目标分布为1-smoothing，将剩余概率分配给词表中的其他单词
# 在训练过程中，使用的平滑指数为0.1
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 损失函数 nn.KLDivLoss / KL散度，又叫做相对熵，算的是两个分布之间的距离，越相似则越接近零。
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        # 如果我们想要修改tensor的数值，但是又不希望被autograd记录，那么我么可以对tensor.data进行操作
        true_dist = x.data.clone()
        # [个人理解！] 减去UNK 和target的idx 之后取均匀分布 smoothing/(n-2)
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 将target的idx在one-hot中位置改成大小为confidence 0.9 (1-smoothing)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 将UNK这一列置0
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# 计算损失
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    # 使得类实例对象可以像调用普通函数一样被调用
    # 相当于 计算loss -> backward -> optimizer.step(). 之后记得梯度清零
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


# 优化器
# Optim wrapper that implements rate
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def seq_padding(X, padding=0):
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    # （注意这里默认padding id是0，相当于是拿<UNK>来做了padding）
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


# 获取中英，word2idx和idx2word字典
def get_word_dict():
    import csv
    cn_idx2word = {}
    cn_word2idx = {}
    en_idx2word = {}
    en_word2idx = {}
    with open("data/word_name_dict/cn_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            cn_idx2word[int(l[0])] = l[1]
            cn_word2idx[l[1]] = int(l[0])
    with open("data/word_name_dict/en_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            en_idx2word[int(l[0])] = l[1]
            en_word2idx[l[1]] = int(l[0])
    return cn_idx2word, cn_word2idx, en_idx2word, en_word2idx


# 保存预测的翻译结果到文件中
def bleu_candidate(sentence):
    with open(BLEU_CANDIDATE, 'a+', encoding='utf-8') as f:
        f.write(sentence + '\n')


# 保存参考译文到文件中
def bleu_references(read_filename, save_filename):
    writer = open(save_filename, 'a+', encoding='utf-8')
    with open(read_filename, 'r', encoding="utf-8") as f_read:
        for line in f_read:
            line = line.strip().split('\t')
            sentence_tap = " ".join([w for w in line[1]])
            writer.write(sentence_tap + '\n')
    writer.close()
    print('写入成功')
