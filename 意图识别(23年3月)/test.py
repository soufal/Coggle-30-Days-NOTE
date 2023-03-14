'''
Author: Soufal
Date: 2023-03-14 15:22:08
Description: 
'''
from gensim.models import Word2Vec
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
my_model = Word2Vec.load('./train_vec1.model')

tq = my_model.wv.most_similar('飞机', topn=5)

print(tq)

