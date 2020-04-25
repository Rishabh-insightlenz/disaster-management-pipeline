#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import pickle


# In[2]:


# load data from database
engine = create_engine('sqlite:///Disaster.db')
df = pd.read_sql('Disasters',con=engine)


# In[3]:


df.head()


# In[4]:


X = df['message']


# In[5]:


Y = df[df.columns[5:]]


# In[6]:


added=pd.get_dummies(df[['related','genre']])
y=pd.concat([Y,added],axis=1)


# ### 2. Write a tokenization function to process your text data

# In[7]:


def tokenize(text):
    # take out all punctuation while tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[8]:


moc = MultiOutputClassifier(RandomForestClassifier())

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', moc)
    ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train.as_matrix(), y_train.as_matrix())
y_pred = pipeline.predict(X_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[10]:


def get_results(y_test, y_pred):
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for cat in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[cat], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    return results


# In[11]:


results=get_results(y_test,y_pred)
results


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[12]:


pipeline.get_params()


# In[13]:


parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

cv = GridSearchCV(pipeline,parameters)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:


cv.fit(X_train.as_matrix(), y_train.as_matrix())
y_pred = cv.predict(X_test)
results2 = get_results(y_test, y_pred)
results2


# In[16]:


cv.best_estimator_


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[17]:


# testing a pure decision tree classifier
moc = MultiOutputClassifier(DecisionTreeClassifier())

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', moc)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train.as_matrix(), y_train.as_matrix())
y_pred = pipeline.predict(X_test)
results = get_results(y_test, y_pred)
results


# ### 9. Export your model as a pickle file

# In[18]:


pickle.dump(cv, open('classifier.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




