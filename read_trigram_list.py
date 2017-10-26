
from nltk.util import ngrams
import numpy as np
import json
import random
import time
import datetime

import pickle

lines = []
for i in range(22):

    print(i)
    txt_file = 'trigram_list/trigram_list_'+str(i)+'.txt'

    
    lines1 = []
    with open(txt_file) as file:
        for line in file:
            line = line.strip() #or some other preprocessing
            lines1.append(line)
    print(len(lines1),len(lines))
    lines = list(set(lines1) | set(lines))

print (len(lines))
thefile = open('all_trigram_list.txt', 'w')
for item in lines:
    thefile.write("%s\n" % item)

'''
        for line in f:
        values = line.split(' ')
        print (values)

        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
'''
