
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

import pymysql.cursors


LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
size = 16200 # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * size # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 5 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.

# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.




def load_model():
    query = Input(shape=(None, WORD_DEPTH))
    pos_doc = Input(shape=(None, WORD_DEPTH))
    neg_docs = [Input(shape=(None, WORD_DEPTH)) for j in range(J)]
    BATCH_SIZE = 50
    filepath = 'weight.h5'
    query_conv = Convolution1D(K, FILTER_LENGTH, padding="same", input_shape=(None, WORD_DEPTH), activation="tanh")(query)
    query_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))(query_conv)  # See section 3.4.
    query_sem = Dense(L, activation="tanh", input_dim=K)(query_max)  # See section 3.5.
    doc_conv = Convolution1D(K, FILTER_LENGTH, padding="same", input_shape=(None, WORD_DEPTH), activation="tanh")
    doc_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))
    doc_sem = Dense(L, activation="tanh", input_dim=K)
    pos_doc_conv = doc_conv(pos_doc)
    neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]
    pos_doc_max = doc_max(pos_doc_conv)
    neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]
    pos_doc_sem = doc_sem(pos_doc_max)
    neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]
    R_Q_D_p = dot([query_sem, pos_doc_sem], axes=1, normalize=True)  # See equation (4).
    R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes=1, normalize=True) for neg_doc_sem in neg_doc_sems]
    concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
    concat_Rs = Reshape((J + 1, 1))(concat_Rs)
    weight = np.array([1]).reshape(1, 1, 1)
    with_gamma = Convolution1D(1, 1, padding="same", input_shape=(J + 1, 1), activation="linear", use_bias=False, weights=[weight])(concat_Rs)  # See equation (5).
    with_gamma = Reshape((J + 1,))(with_gamma)
    prob = Activation("softmax")(with_gamma)  # See equation (5).
    model = Model(inputs=[query, pos_doc] + neg_docs, outputs=prob)
    model.compile(optimizer="adadelta", loss="categorical_crossentropy")
    model.load_weights(filepath)
    return model

def build_trigram():
    trigram_list = []
    all_trigram_list_txt_file = 'all_trigram_list.txt'
    with open(all_trigram_list_txt_file) as trigram_file:
        for line in trigram_file:
            line = line.strip()  # or some other preprocessing
            trigram_list.append(line)

    trigram_index = {}
    hehe = 0
    for i in trigram_list:
        trigram_index[i] = hehe
        hehe += 1
    print("Finishing trigram index", len(trigram_index))
    trigram_list = []
    return trigram_index

def generate_tags(string):
    new_string = string.replace('<','')
    new_string = new_string.replace('>',' ')
    str_list = new_string.split()
    return str_list




model = load_model()
trigram_index = build_trigram()


config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '0472',
    'db': 'testdb',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,

}


connection = pymysql.connect(**config)

try:
    with connection.cursor() as cursor:


        all_query = []
        all_groud_truth = []
        all_tags = []
        with open('testing1000_1.json', encoding='utf-8') as jsondata:
            file = json.load(jsondata)
            for row in file:

                all_query.append(row['question1'])
                all_groud_truth.append(row['question2'])
                all_tags.append(row['tagname1'])

        precision1 = 0
        precision5 = 0
        precision10 = 0
        map_sum = 0

        for this in range(len(all_query)):

            if this % 50 == 0 :
                print("This is query %d" % this)

                if this!= 0 :
                    print(precision1 / this, (precision1+precision5) / this, (precision1+precision5+precision10) / this)
                    print(map_sum/this)
            tags = generate_tags(all_tags[this])

            query = all_query[this]
            groud_truth = all_groud_truth[this]

            if len(tags) == 2:
                sql1 = "select title from mainsite_questions where tags like \'%" + tags[0] + '%\' and tags like \'%' + \
                       tags[1] + '%\' limit 100;'
            elif len(tags) == 3:
                sql1 = "select title from mainsite_questions where tags like \'%" + tags[0] + '%\' and tags like \'%' + \
                       tags[1] + '%\' and tags like \'%' + tags[2] + '%\' limit 100;'
            elif len(tags) == 4:
                sql1 = "select title from mainsite_questions where tags like \'%" + tags[0] + '%\' and tags like \'%' + \
                       tags[1] + '%\' and tags like \'%' + tags[2] + '%\' and tags like \'%' + tags[
                           3] + '%\' limit 100;'
            elif len(tags) >= 5:
                sql1 = "select title from mainsite_questions where tags like \'%" + tags[0] + '%\' and tags like \'%' + \
                       tags[1] + '%\' and tags like \'%' + tags[2] + '%\' and tags like \'%' + tags[
                           3] + '%\' and tags like \'%' + tags[4] + '%\' limit 100;'
            #print(sql1)
            cursor.execute(sql1)
            result1 = cursor.fetchall()
            #print (len(result1))


            query = text_to_word_sequence(query)
            groud_truth = text_to_word_sequence(groud_truth)

            neg_s = []
            for num in range(len(result1)):
                this_question = result1[num]['title']
                #print(this_question)
                neg_s.append(text_to_word_sequence(this_question))
            #print(len(neg_s))
            this_query = query

            l_Q = np.zeros( (1, len(this_query) , WORD_DEPTH) , dtype='float32')
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
                    # print(this_trigram_list[m])
                    # print(trigram_index[this_trigram_list[m]])
                    if this_trigram_list[m] in trigram_index:
                        l_Q[0][n][trigram_index[this_trigram_list[m]]] += 1
                    else:
                        pass
                        #print("Unkown trigram!", this_trigram_list[m])
                l_Q[0][n] = l_Q[0][n] / np.linalg.norm(l_Q[0][n])

            this_query = groud_truth

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
                    # print(this_trigram_list[m])
                    # print(trigram_index[this_trigram_list[m]])
                    if this_trigram_list[m] in trigram_index:
                        l_P[0][n][ trigram_index[this_trigram_list[m]] ] += 1
                    else:
                        pass
                        #print("Unkown trigram!", this_trigram_list[m])
                l_P[0][n] = l_P[0][n] / np.linalg.norm(l_P[0][n])


            l_Ns = []
            for i in range(len(neg_s)):
                this_query = neg_s[i]

                l_N = np.zeros((1, len(this_query), WORD_DEPTH), dtype='float32')
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
                        # print(this_trigram_list[m])
                        # print(trigram_index[this_trigram_list[m]])
                        if this_trigram_list[m] in trigram_index:
                            l_N[0][n][trigram_index[this_trigram_list[m]]] += 1
                        else:
                            pass
                            #print("Unkown trigram!", this_trigram_list[m])
                    l_N[0][n] = l_N[0][n] / np.linalg.norm(l_N[0][n])
                l_Ns.append(l_N)
            #print("Start predicting!")
            sum_pos = 0
            if len(l_Ns) < 100:
                range_num = int(len(l_Ns)/5)
            else:
                range_num = 20
            for batch in range(range_num):
                #print(batch*5 ,batch*5 +J)
                history = model.predict([l_Q, l_P] + [l_Ns[j] for j in range(batch*5 ,batch*5 +J)] , verbose=0)
                history = history[0]
                this_history = []
                for aaa in range(J):
                    this_history.append(history[aaa])
                #print(this_history)
                g_t = this_history[0]
                this_history.sort(reverse =True)
                sum_pos += this_history.index(g_t)
                #print(this_history)
                this_history = []
                history = []

            sum_pos += 1
            print(sum_pos)
            if sum_pos <= 1:
                precision1 += 1
            elif sum_pos >=2 and sum_pos <= 5:
                precision5 += 1
            elif sum_pos>= 6 and sum_pos <= 10:
                precision10 += 1
            else:
                pass

            map_sum += 1 / sum_pos

            l_Q = []
            l_P = []
            neg_s = []
            l_N = []
            l_Ns = []


finally:
    connection.close()






