#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd  
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report 


# In[2]:


# Reading the csv file
cardio_f = pd.read_csv("cardio_train.csv")
cardio_f.head()


# In[5]:


# Removing id & cardio
feature_f = cardio_f [['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

x = np.asarray(feature_f)

y = np.asarray(cardio_f['cardio'])



# In[7]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.9, random_state = 4)

'''
Train sample: 6930
Test sample: 62371
'''


# In[8]:


# Creating the model
classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C = 1)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)


# In[12]:
 

print('Predicted labels: ', y_predict)
print('Accuracy: ', accuracy_score(y_test, y_predict))


# In[ ]:

pickle.dump(cardio_f, open('finalized_model.sav', 'wb'))

