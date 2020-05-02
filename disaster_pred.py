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
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder




def predict_keyword(new_df):
    if 'target' in new_df.columns:
        test= new_df[new_df['keyword'].isnull()]
        y_test= test['target']
        test.drop(['keyword','target'],axis=1, inplace=True)

        train= new_df[new_df['keyword'].notnull()]
        y_train= train['target']
        y= train['keyword']
        train.drop(['keyword','target'],axis=1, inplace=True)

    else:
        test= new_df[new_df['keyword'].isnull()]
        test.drop(['keyword'],axis=1, inplace=True)

        train= new_df[new_df['keyword'].notnull()]
        y= train['keyword']
        train.drop(['keyword'],axis=1, inplace=True)


    X= train
    model = LogisticRegression(solver = 'lbfgs')

    model.fit(X,y)
    keywords= model.predict(test)
    test['keyword']= keywords
    train['keyword']=y
    if 'target' in new_df.columns:
        test['target']= y_test
        train['target']= y_train
    new_df= pd.concat([test,train])
    return new_df




def preprocess_data(sentences):
    lemmatizer= WordNetLemmatizer()

    for i in range(len(sentences)):
        words= nltk.word_tokenize(sentences[i])
        words= [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        sentences[i]= " ".join(words)

    return sentences

df= pd.read_csv(r'train.csv')

#df.dropna(subset=['keyword'], inplace=True)


#Sentences Pickle
#sentences= preprocess_data(df['text'])
# =============================================================================
# pickle_out = open("sentences2.pickle","wb")
# pickle.dump(sentences, pickle_out)
# pickle_out.close()
# =============================================================================

pickle_in = open("sentences.pickle","rb")
sentences = pickle.load(pickle_in)






cv= TfidfVectorizer()
X= cv.fit_transform(sentences).toarray()
y= df['target']

new_df= pd.DataFrame(data=X)
new_df['keyword']=df['keyword']
new_df['id']=df['id']
new_df['target']= y
new_df= predict_keyword(new_df)
#y= new_df['target']
#new_df.drop(['target'], axis=1, inplace=True)



#x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= .2)



# =============================================================================
# model= xg_reg = xgb.XGBClassifier(subsample= 1.0,
#                                  min_child_weight= 10,
#                                  learning_rate= 0.1,
#                                  gamma= 1.5,
#                                  booster= 'gbtree',
#                                  colsample_bytree= 1.0)
# =============================================================================
#model= LogisticRegression()
model= GaussianNB()
#y= new_df['target']
le= LabelEncoder()
new_df['keyword']= le.fit_transform(new_df['keyword'])

y= new_df['target']
new_df.drop(['target'], axis=1, inplace=True)
model.fit(new_df, y)

#model.score(x_test, y_test)



df2= pd.read_csv('test.csv')
df2.reset_index(drop=True, inplace=True)
#sentences=  preprocess_data(df2['text'])
pickle_in = open("sentences2.pickle","rb")
sentences = pickle.load(pickle_in)


new_X= cv.transform(sentences).toarray()
new_df2= pd.DataFrame(data=new_X)
new_df2['keyword']=df2['keyword']
new_df2['id']= df2['id']
new_df2= predict_keyword(new_df2)

new_df2['keyword']= le.fit_transform(new_df2['keyword'])

y_predict= model.predict(new_df2)
new_df2['target']= y_predict

df3= new_df2[['id','target']]
df3.to_csv('final.csv', index=False)


