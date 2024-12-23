# =============================================================================
# import numpy as np
# 
# c = np.array([1, 0, 0, 0, 0, 0, 0]) #입력
# W = np.random.randn(7, 3)           #가중치
# h = np.matmul(c,W)                  #중간 노드
# print(h)                            #[[-0.70012195 0.25204755 -0.79774592]]
# 
# 
# =============================================================================

# =============================================================================
# 
# import sys
# sys.path.append('..')
# from common.util import preprocess
# 
# text = 'You say goodbye and I say hello.'
# corpus, word_to_id, id_to_word= preprocess(text)
# print(corpus)       # [0 1 2 3 4 1 5 6]
# 
# print(id_to_word)   
# =============================================================================


# =============================================================================
# def create_contexts_target(corpus, window_size = 1):
#     target = corpus[window_size:-window_size]
#     contexts = []
#     
#     for idx in range(window_size,len(corpus)-window_size):
#         cs = []
#         for t in range(-window_size,window_size+1):
#             if t == 0:
#                 continue
#             cs.append(corpus[idx + t])
#         contexts.append(cs)    
#     
#     return np.array(contexts), np.array(target)
#         
# contexts, target = create_contexts_target(corpus,window_size=1)
# 
# print(contexts)
# print(target)
# =============================================================================
# =============================================================================
# 
# import sys
# sys.path.append('..') 
# from common.util import preprocess,create_contexts_target,convert_one_hot
# 
# text = 'You say goodbye and I say hello.'
# corpus, word_to_id, id_to_word = preprocess(text)
# 
# contexts, target = create_contexts_target(corpus, window_size=1)
# 
# vocab_size = len(word_to_id)
# target = convert_one_hot(target, vocab_size)
# contexts = convert_one_hot(contexts, vocab_size)
# =============================================================================

import matplotlib.font_manager as fm
fm.fontManager.ttflist 
[f.name for f in fm.fontManager.ttflist]

import matplotlib.pyplot as plt

# 가능한 font list 확인
import matplotlib.font_manager as fm
f = [f.name for f in fm.fontManager.ttflist]
print(f)
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
