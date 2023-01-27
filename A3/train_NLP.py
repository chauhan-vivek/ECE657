# import required packages
import re
import os
import pandas as pd
import numpy as np
import glob
import nltk
import pickle
import string

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.utils import shuffle

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def dataset(filepath_pos, filepath_neg):
    
    positive_files = glob.glob(filepath_pos)
    negative_files = glob.glob(filepath_neg)
    
    positive_reviews, negative_reviews = [], []
    
    for pos in positive_files:
        with open(pos,  'r', encoding='utf-8') as file_pos:
            positive_reviews.append(file_pos.readline()) 

    for neg in negative_files:
        with open(neg,  'r', encoding='utf-8') as file_neg:
            negative_reviews.append(file_neg.readline())  
            
    len(positive_reviews)
    positive_labels = np.ones(len(positive_reviews)).tolist()
    negative_labels = np.zeros(len(negative_reviews)).tolist()
    
    reviews = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels
    
    
    tokenized_reviews = []
    for review in reviews:
        review = review.lower()
        review = re.sub("\\s", " ", review)
        review = re.sub("[^a-zA-Z' ]", "", review)
        tokens = review.split(' ')
        #url_pattern = r'[A-Za-z0-9]+://[A-Za-z0-9%-_]+(/[A-Za-z0-9%-_])*(#|\\?)[A-Za-z0-9%-_&=]*'
        #pattern = re.compile(url_pattern)
        #review = pattern.sub('',review)
        #punc_pattern = string.punctuation
        #punc = r"[{}]".format(punc_pattern)
        #review= re.sub(punc, "", review)
        
        #regexp_tokenizer = RegexpTokenizer(r'\w+')
        #tokens = regexp_tokenizer.tokenize(review)

        stopword = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopword]
        
        filtered_words = ' '.join(filtered_tokens)
        tokenized_reviews.append(filtered_words)
    
    tokenized_reviews, labels = shuffle(tokenized_reviews, labels)
        
    return tokenized_reviews, labels
 
if __name__ == "__main__":
    filepath_pos = r'data\aclImdb\train\pos\*.txt'
    filepath_neg = r'data\aclImdb\train\neg\*.txt'
    reviews, labels = dataset(filepath_pos, filepath_neg)
    
    opt_dim = 50 #output_dimensions for embedding layer (number of dimensions of vector)
    
    percentile = int(np.percentile([len(seq) for seq in reviews], 80)) # for maxlen of per review to consider
    
    truncatedData = [''.join(seq[:percentile]) for seq in reviews]
    
    tokenizer = Tokenizer(10000)
    tokenizer.fit_on_texts(truncatedData)
    final_data = tokenizer.texts_to_sequences(truncatedData)
    
    final_data = pad_sequences(final_data, maxlen=percentile, padding='post')
    pickle.dump(tokenizer, open(r"data\token.p", "wb"))
    
    
    vocabSize = len(tokenizer.word_index)
    labels = np.array(labels).reshape((-1,1))
    labels = to_categorical(labels, num_classes=2)
    
    model = Sequential()
    emb = Embedding(input_dim=vocabSize+1, output_dim=opt_dim,
                    input_length=percentile)
    model.add(emb)
    model.add(Conv1D(64, 3, padding='same', activation='tanh'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(units=2, activation='sigmoid'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(final_data, labels, epochs=10, verbose = True)
    
    model.save("models/20955627_NLP_model.model")
    
    evaluate = model.evaluate(final_data, labels)
    print("Train Accuracy and Loss is: ", str(evaluate[1] * 100), "% and ", evaluate[0], " respectively.", sep = '')