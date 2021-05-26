import numpy as np
from numpy import dot

voc_file = np.load('./data/vocab_iam.txt',allow_pickle=True)
print(voc_file[:100])

# f = np.load('nnlm_word_embeddings.zh.npy')
f=np.load("nnlm_word_embeddings.zh.npy",allow_pickle=True)
print(len(voc_file) == len(f))


v_1 = f[0]
v_2 = f[1]
v_3 = f[2]


def get_s(v_1, v_2):
    v1_norm = sum([i*i for i in v_1]) ** 0.5
    v2_norm = sum([i*i for i in v_2]) ** 0.5
    abs_num = abs(dot(v_1, v_2))
    return abs_num / (v1_norm * v2_norm)

# 駱驛
# 東返
