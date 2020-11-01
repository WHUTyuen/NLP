# print(chr(68))
import torch
import torchtext
from torchtext import vocab

gv = torchtext.vocab.GloVe(name="6B",dim=50)
# print(len(gv.vectors))
# print(gv.vectors.shape)
#
print(gv.stoi['tokyo'])
print(gv.vectors[1363])
#
# print(gv.itos[1363])

def get_wv(word):
    return gv.vectors[gv.stoi[word]]


#取出这个词向量，拿这个向量去遍历所有的向量，求距离，拿出10个最近的词
def sim_10(word,n = 10):
    aLL_dists = [(gv.itos[i],torch.dist(word,w)) for i,w in enumerate(gv.vectors)]
    return sorted(aLL_dists,key=lambda t: t[1])[:n]

def answer(w1,w2,w3):
    print("[%s:%s==%s:?]"%(w1,w2,w3))
    #w1-w2 = w3-w4
    #w4 = w3-w1+w2
    w4 = get_wv(w3)-get_wv(w1)+get_wv(w2)
    return sim_10(w4)[0]
print(answer("beijing","china","tokyo"))