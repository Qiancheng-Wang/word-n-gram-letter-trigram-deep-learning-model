from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

from nltk.util import ngrams
import numpy as np
import json
import random
import time
import datetime

import pickle




LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
size = 24245 # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * size # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 2 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

def trigram_generater(location,file_name):

    question1 = []
    question2 = []
    questions1 = []
    questions2 = []
    questions =[]
    is_duplicate = []
    with open(file_name, encoding='utf-8') as jsondata:
        file = json.load(jsondata)
        for row in file:
            if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
                pass
            else:
                question1.append(row['question1'])
                question2.append(row['question2'])
                is_duplicate.append(row['is_duplicate'])

    print('Question pairs: %d' % len(question1))


        # Build tokenized word index
    for i in range(len(question1)):
        questions1.append(text_to_word_sequence(question1[i]))
        questions2.append(text_to_word_sequence(question2[i]))

    questions = questions1 + questions2

    trigram_list = find_trigrams(questions)


    #write list to a file
    thefile = open('trigram_list/trigram_list_'+str(location)+'.txt', 'w')
    for item in trigram_list:
        thefile.write("%s\n" % item)
    #thefile.write(str(trigram_list))
    print("This trigram list length is",len(trigram_list))
    print("end time is : ",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))


def find_trigrams(input_list):
    trigram_list = []
    for j in range(len(input_list)):
        if j % 5000 == 0: print("Input sentence",j)
        for i in range(len(input_list[j])):
            if len(input_list[j][i]) <= 2:
                test = input_list[j][i]
                trigram_list.append(test)
            for k in range(len(input_list[j][i])-2):
                test = input_list[j][i][k:(k+3)]
                if test in trigram_list:
                    pass
                else:
                    trigram_list.append(test)
    trigram_list = list(set(trigram_list))
    return trigram_list


for i in range(22):
    json_file_name = 'raw/raw'+str(i)+'.json'
    print("start time is : ", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    trigram_generater(i,json_file_name)
