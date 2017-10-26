from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model

from nltk.util import ngrams
import numpy as np
import json
import random


import pickle

LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
size = 16200 # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * size # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.
query = Input(shape = (None, WORD_DEPTH))
pos_doc = Input(shape = (None, WORD_DEPTH))
neg_docs = [Input(shape = (None, WORD_DEPTH)) for j in range(J)]
BATCH_SIZE = 50

filepath = 'weight.h5'





query_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, WORD_DEPTH), activation = "tanh")(query) # See equation (2).

query_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))(query_conv) # See section 3.4.

# In this step, we generate the semantic vector represenation of the query. This
# is a standard neural network dense layer, i.e., y = tanh(W_s v + b_s). Again,
# the paper does not include bias units.
query_sem = Dense(L, activation = "tanh", input_dim = K)(query_max) # See section 3.5.

# The document equivalent of the above query model.
doc_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, WORD_DEPTH), activation = "tanh")
doc_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))
doc_sem = Dense(L, activation = "tanh", input_dim = K)

pos_doc_conv = doc_conv(pos_doc)
neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

pos_doc_max = doc_max(pos_doc_conv)
neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

pos_doc_sem = doc_sem(pos_doc_max)
neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

# This layer calculates the cosine similarity between the semantic representations of
# a query and a document.
R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) # See equation (4).
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems] # See equation (4).

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
concat_Rs = Reshape((J + 1, 1))(concat_Rs)

# In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
# described as a smoothing factor for the softmax function, and it's set empirically
# on a held-out data set. We're going to learn gamma's value by pretending it's
# a single 1 x 1 kernel.
weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs) # See equation (5).
with_gamma = Reshape((J + 1, ))(with_gamma)

# Finally, we use the softmax function to calculate P(D+|Q).
prob = Activation("softmax")(with_gamma) # See equation (5).

# We now have everything we need to define our model.
model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy")

Nega = 'This is a negative post and should not be similar with anybody'

# Build tokenized word index
trigram_list = []
all_trigram_list_txt_file = 'all_trigram_list.txt'
with open(all_trigram_list_txt_file) as trigram_file:
    for line in trigram_file:
        line = line.strip()  # or some other preprocessing
        trigram_list.append(line)

print("Finishing trigram list loding",len(trigram_list))
trigram_index = {}

#word->index
hehe = 0
for i in trigram_list:
    trigram_index[i] = hehe
    hehe += 1
print("Finishing trigram index",len(trigram_index))
trigram_list = []

for epoch in range(10):

    for raw_i in range(20):
        print("Epoch %d, Rawdata %d" % (epoch,raw_i))
        question1 = []
        question2 = []
        questions1 = []
        questions2 = []
        questions =[]
        with open('raw/raw'+str(raw_i)+'.json', encoding='utf-8') as jsondata:
            file = json.load(jsondata)
            for row in file:
                if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
                    pass
                else:
                    question1.append(row['question1'])
                    question2.append(row['question2'])

        #print('Question pairs: %d' % len(question1))

        for i in range(len(question1)):
            questions1.append(text_to_word_sequence(question1[i]))
            questions2.append(text_to_word_sequence(question2[i]))

        question1 = []
        question2 = []

        l_Qs = []
        l_Ps = []
        for i in range(len(questions1)):
            if i % 5000 == 0:print("Processing the pair %s ",i," Questions")

            this_query = questions1[i]

            l_Q = np.zeros( (1, len(this_query) , WORD_DEPTH) , dtype='float32')
            for n in range(len(this_query)):
                this_query_split = this_query[n:(n+3)]

                this_trigram_list = []
                for j in range(len(this_query_split)):
                    if len(this_query_split[j]) <= 2:
                            test = this_query_split[j]
                            if test in this_trigram_list:
                                pass
                            else:
                                this_trigram_list.append(test)
                    else:
                        for k in range(len(this_query_split[j])-2):
                            test = this_query_split[j][k:(k+3)]
                            if test in this_trigram_list:
                                pass
                            else:
                                this_trigram_list.append(test)
                #print(this_trigram_list)
                this_trigram_list = list(set(this_trigram_list))
                for m in range(len(this_trigram_list)):
                    #print(this_trigram_list[m])
                    #print(trigram_index[this_trigram_list[m]])
                    if this_trigram_list[m] in trigram_index:
                        l_Q[0 ][n][ trigram_index[this_trigram_list[m]]  ] += 1
                    else:
                        print("Unkown trigram!",this_trigram_list[m])
                l_Q[0][n] = l_Q[0][n] / np.linalg.norm(l_Q[0][n])
            l_Qs.append(l_Q)

            this_query = questions2[i]

            l_P = np.zeros((1, len(this_query), WORD_DEPTH), dtype='float32')
            for n in range(len(this_query)):
                this_query_split = this_query[n:(n + 3)]

                this_trigram_list = []
                for j in range(len(this_query_split)):
                    if len(this_query_split[j]) <= 2:
                        test = this_query_split[j]
                        if test in this_trigram_list:
                            pass
                        else:
                            this_trigram_list.append(test)
                    else:
                        for k in range(len(this_query_split[j]) - 2):
                            test = this_query_split[j][k:(k + 3)]
                            if test in this_trigram_list:
                                pass
                            else:
                                this_trigram_list.append(test)
                # print(this_trigram_list)
                this_trigram_list = list(set(this_trigram_list))
                for m in range(len(this_trigram_list)):
                    if this_trigram_list[m] in trigram_index:
                        l_P[0][n][ trigram_index[this_trigram_list[m]] ] += 1
                    else:
                        print("Unkown trigram!",this_trigram_list[m])
                l_P[0][n] =  l_P[0][n] / np.linalg.norm(l_P[0][n])
            l_Ps.append(l_P)

        #print("Finishing normalization")


        l_Ns = [[] for j in range(J)]
        for i in range(len(questions1)):
            if i % 5000 == 0:print("Processing the negative %s ",i," Question")
            possibilities = list(range(len(questions1)))
            possibilities.remove(i)
            num = random.sample(possibilities, J)
            #print(num)
            for j in range(J):
                negative = num[j]
                l_Ns[j].append(l_Ps[negative])
        '''
        Nega_vec = np.zeros( (1, len(Nega) , WORD_DEPTH) ,, dtype=float32)
        for n in range(len(Nega)):
            this_query_split = Nega[n:(n+3)]

            this_trigram_list = []
            for j in range(len(this_query_split)):
                if len(this_query_split[j]) <= 2:
                        test = this_query_split[j]
                        if test in this_trigram_list:
                            pass
                        else:
                            this_trigram_list.append(test)
                else:
                    for k in range(len(this_query_split[j])-2):
                        test = this_query_split[j][k:(k+3)]
                        if test in this_trigram_list:
                            pass
                        else:
                            this_trigram_list.append(test)
                #print(this_trigram_list)
            this_trigram_list = list(set(this_trigram_list))
            for m in range(len(this_trigram_list)):
                #print(this_trigram_list[m])
                    #print(trigram_index[this_trigram_list[m]])
                if this_trigram_list[m] in trigram_index:
                    Nega_vec[0][n][ trigram_index[this_trigram_list[m]]  ] += 1
                else:
                    print("Unkown trigram!",this_trigram_list[m])
            Nega_vec[0][n] = Nega_vec[0][n] / np.linalg.norm(Nega_vec[0][n])

        l_Ns = [[] for j in range(J)]
        for i in range(len(questions1)):
            if i % 2000 == 0:print("Processing the negative %s ",i," Question")
            for j in range(J):

                l_Ns[j].append(Nega_vec)
        '''
        #label
        y = np.zeros(J + 1).reshape(1, J + 1)
        y[0, 0] = 1

        print("Finishing Processing Data!\n\n")
        print("Start training\n")

        for i in range(len(questions1)):
            if i % 5000 == 0:print(epoch,"   The ",i," one in total",len(questions1),"ones")
            history = model.fit([l_Qs[i], l_Ps[i]] + [l_Ns[j][i] for j in range(J)], y, epochs = 0, verbose = 0)
        y = []
        question1 = []
        question2 = []
        questions1 = []
        questions2 = []
        questions = []
        l_Qs = []
        l_Ps = []
        l_Ns = []

model.save(filepath)
'''
print("start test")

accuracy = 0
test_number = len(questions1)
for i in range(20):
    if i % 2000 == 0:print(i)
    history = model.predict([l_Qs[i], l_Ps[i]] + [l_Ns[j][i] for j in range(J)], verbose = 1)
    if history[0][0] >= 0.5:
        accuracy += 1
    print(history[0][0])

print("This predict accuracy is :", accuracy/len(questions1) )

    
accuracy = 0
for i in range(9980,10000):
    if i % 2000 == 0:print(i)
    history = model.predict([l_Qs[i], l_Ps[i]] + [l_Ns[j][i] for j in range(J)], verbose = 1)
    if history[0][0] >= 0.5:
        accuracy += 1
    print(history[0][0])

print("This predict accuracy is :", accuracy/20 )

'''
