import argparse

import numpy as np
from nltk import word_tokenize
from torch.autograd import Variable

from model import make_model
from settings import *
from utils import get_word_dict,subsequent_mask

"""
单个句子输入，单个句子翻译输出
"""

# 初始化模型
model = make_model(SRC_VOCAB,TGT_VOCAB,LAYERS,D_MODEL,D_FF,H_NUM,DROPOUT)
cn_idx2word, cn_word2idx, en_idx2word, en_word2idx = get_word_dict()
model.load_state_dict(torch.load(SAVE_FILE,map_location=torch.device('cpu')))

# 将句子单词转为id表示
def sentence2id(sentence):
    en = []
    en.append(['BOS'] + word_tokenize(sentence.lower()) + ['EOS'])
    sentence_id = [[int(en_word2idx.get(w,0)) for w in e] for e in en]
    return sentence_id

# 将句子id列表转换为tensor，并且生成输入的mask矩阵
def src_handle(X):
    src = torch.from_numpy(np.array(X)).long().to(DEVICE)
    src_mask = (src != 0).unsqueeze(-2)
    return src, src_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def output(out):
    translation = []
    # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
    for j in range(1, out.size(1)): # 生成的最大程度的序列
        # 获取当前下标的输出字符
        sym = cn_idx2word[out[0, j].item()]
        # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
        if sym != 'EOS':
            translation.append(sym)
        else:
            break
    # 打印模型翻译输出的中文句子结果
    # print("translation: %s" % " ".join(translation))
    return ''.join(translation)

# 实现机器翻译
def machine_translate(sentence):
    src,src_mask = src_handle(sentence2id(sentence))
    out = greedy_decode(model,src,src_mask,max_len=50,start_symbol=int(cn_word2idx.get('BOS')))
    cn_result = output(out)
    return cn_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True)
    args = parser.parse_args()
    print('result : ',machine_translate(args.sentence))