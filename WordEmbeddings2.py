#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import pickle 

import gzip
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

dataframe = getDF('reviews_Musical_Instruments_5.json.gz')
    
'''
#making a raw unproccessed corpus for easier extraction\

Y_actual = dataframe.iloc[:, 5].values
for i in range(len(Y_actual)):
    if Y_actual[i] >= 3:
        Y_actual[i] = 1
    else :
        Y_actual[i] = 0

corpus  = []
for i in range(0, len(dataframe)):
    review = re.sub('[^a-zA-Z]', ' ', dataframe['reviewText'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)
corpus[2927] = 'good'
corpus[4401] = 'good'
corpus[8739] = 'good'
corpus[9175] = 'good'
corpus[9306] = 'good'   
corpus[9313] = 'good' 
corpus[9342] = 'good'  

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

    
corpus_temp  = []
corpus_split  = []
for i in range(len(corpus)):
    corpus_temp  = corpus[i].split()
    corpus_split .append(corpus_temp)

import gensim
#model = gensim.models.Word2Vec(corpus_split, min_count = 1)
model = gensim.models.KeyedVectors.load_word2vec_format('C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\gensim\\models\\GoogleNews-vectors-negative300.bin', binary = True)

#Using goolge's data to convert corpus into vectors with 300 parameters
#for loop: take each row , then take each word in the row and convert, then average 
#of each row per parameter so each tweet has 300 parameter instead of having mult-
#iple words with 300 parameter
corpus_numbers = []
    
for i in range(len(corpus)):
    actual_words = 0    
    average = [0]*300
    for some_word in corpus_split[i]:
        word  = model[some_word]
        if word  != "nan":
            actual_words += 1 
            for k in range(300):                
                number_of_terms = len(corpus_split[i])     
                average[k] += word[k]
                    
    for flag  in range(300) :
        average [flag] = average [flag]/actual_words
    
    corpus_numbers.append(average)
#convert to np.array   
corpus_final = np.array(corpus_numbers)

 
pickle.dump(corpus_final, open('Preprocessed_data','wb'))
pickle.dump(Y_actual, open('Preprocessed_data_sentiment', 'wb'))
'''
corpus_final  = pickle.load(open('Preprocessed_data','rb'))
Y_actual = pickle.load( open('Preprocessed_data_sentiment', 'rb'))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(corpus_final, Y_actual, test_size = 0.25, random_state = 0)
'''
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 151, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train , batch_size = 10, epochs = 100)
'''

classifier = pickle.load(open('Musical_improved', 'rb'))
#pickle.dump(classifier, open('MusicalInstruments', 'wb'))
#predicting the result and ensuring that its binary
Y_pred = classifier.predict(X_test)
for i in range(len(Y_pred)):
    if Y_pred[i] >= 0.5:
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0
        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
'''
#converting results into a dataframe for external use
dataset_final = pd.DataFrame(Y)
dataset_final.to_csv('answers.csv')
'''
    