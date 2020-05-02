#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:20:33 2020

@author: rahul
"""

import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import seaborn as sns

def preprocess_data(sentences):
    lemmatizer= WordNetLemmatizer()

    for i in range(len(sentences)):
        words= nltk.word_tokenize(sentences[i])
        words= [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        sentences[i]= " ".join(words)

    return sentences

df= pd.read_csv(r'train.csv')

#df.dropna(subset=['keyword'], inplace=True)


sentences= preprocess_data(df['text'])
from sklearn.feature_extraction.text import TfidfVectorizer
cv= TfidfVectorizer()
X= cv.fit_transform(sentences).toarray()
y= df['target']





#x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= .2)


from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
model= xg_reg = xgb.XGBClassifier(subsample= 1.0,
                                 min_child_weight= 10,
                                 learning_rate= 0.1,
                                 gamma= 1.5,
                                 booster= 'gbtree',
                                 colsample_bytree= 1.0)
#model= GaussianNB()
model.fit(X, y)

#model.score(x_test, y_test)



df2= pd.read_csv('test.csv')
#df2.dropna(subset=['keyword'], inplace=True)
df2.reset_index(drop=True, inplace=True)
sentences=  preprocess_data(df2['text'])

new_X= cv.transform(sentences).toarray()
y_predict= model.predict(new_X)
df2['target']= y_predict

df3= df2[['id','target']]
df3.to_csv('final.csv', index=False)


