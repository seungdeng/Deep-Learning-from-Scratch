import numpy as np


text = 'You say goodbye and I say Hello.'

text = text.lower()
text = text.replace('.', ' .')
# =============================================================================
# print(text)
# =============================================================================
words = text.split(' ')
# =============================================================================
# print(words)
# =============================================================================

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word


corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
   
    return corpus,word_to_id,id_to_word


import numpy as np
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0],
    ],dtype = np.int32)

print([C[0]])


def create_co_matrix(corpus,vocab_size,window_size = 1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1,window_size + 1):
            left_idx = idx-1
            right_idx = idx+1
            
            if left_idx >=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id,left_word_id] +=1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] +=1
                
    return co_matrix



def cos_similarity(x,y,eps = 1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps) # x,y의 정규화
    return np.dot(nx,ny)



def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    #1. 검색어를 꺼낸다
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    #2. 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
    #3. 코사인 유사도를 기준으로 내림차순 출력
    count = 0
    for i in range(-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i],similarity[i]))
        
        count +=1
        
        if count >= top:
            return
        
        
x = np.array([100,-20,2])
x. argsort() # array([1, 2, 0])
(-x).argsort() # array([0, 2, 1])

def ppmi(C, verbase = False, eps=1e-8):
    M = np.zeros_like(C,dtype = np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[j]*S[i]) + eps)
            M[i,j] = max(0,pmi)
            
            if verbase:
                cnt +=1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
                    
    return M