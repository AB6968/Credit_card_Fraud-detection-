#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("creditcard.csv")


# In[8]:


df


# In[9]:


df.shape


# In[10]:


df.info()

# no missing values


# In[11]:


df.describe()


# In[14]:


df['Class'].unique()


# In[7]:


df['Class'].value_counts()
# data is imbalanced


# Data visualization

# In[15]:


import matplotlib.pyplot as plt


# In[20]:


df.hist(figsize=[30,35])
plt.show()


# In[21]:


df.hist("Class")


# In[27]:


df.boxplot(figsize=(25,7),grid=False)


# In[28]:


import seaborn as sns


# In[33]:


sns.boxplot(df['Amount'])


# In[34]:


sns.pairplot(df)


# In[35]:


#Scale amount by Standardization
from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
df['amount_scaled'] = ss.fit_transform(df['Amount'].values.reshape(-1,1))


# In[36]:


df


# In[37]:


sns.boxplot(df['amount_scaled'])


# In[38]:


df.boxplot(figsize=(25,7),grid=False)


# In[39]:


df


# In[42]:


# split into feature variable
x=df.drop(["Amount", "Class"], axis=1)
x


# In[43]:


# split into response variable
y = df["Class"]
y


# Model Fitting

# In[62]:


get_ipython().system('pip install imblearn')


# In[63]:


# import libraries 
# splitting data 
from sklearn.model_selection import train_test_split

# to check performance of metrics
from sklearn import metrics


# In[64]:


# balancing data
from imblearn.under_sampling import RandomUnderSampler


# In[56]:


get_ipython().system('pip3 install -U scikit-learn')


# In[66]:


# split training and test data
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3, shuffle=True)


# Classification Algorithms

# In[84]:


names=[]
aucs_tests = []
accuracy_tests = []
precision_tests = []
recall_tests = []
f1_score_tests = []

def performance(model):
    for name, model, X_train, y_train, X_test, y_test in model:
        
        #appending name
        names.append(name)
        
        # Build model
        model.fit(X_train, y_train)
        
        #predictions
        y_test_pred = model.predict(X_test)
        
        # calculate accuracy
        Accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
        accuracy_tests.append(Accuracy_test)
        
        # calculate auc
        Aucs_test = metrics.roc_auc_score(y_test , y_test_pred)
        aucs_tests.append(Aucs_test)
        
        #precision_calculation
        Precision_score_test = metrics.precision_score(y_test , y_test_pred)
        precision_tests.append(Precision_score_test)
        
        # calculate recall
        Recall_score_test = metrics.recall_score(y_test , y_test_pred)
        recall_tests.append(Recall_score_test)
        
        #calculating F1
        F1Score_test = metrics.f1_score(y_test , y_test_pred)
        f1_score_tests.append(F1Score_test)
        
        # draw confusion matrix
        cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
        
        print("Model Name :", name)
        print(f"Test Accuracy: {Accuracy_test}")
        print(f"Test AUC: {Aucs_test}")
        print(f"Test Precision: {Precision_score_test}")
        print(f"Test Recall: {Recall_score_test}")
        print(f"Test F1: {F1Score_test}")
        print(f"Confusion matrix: \n {cnf_matrix}")
        print("\n")

        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
        auc = metrics.roc_auc_score(y_test, y_test_pred)
        plt.plot(fpr,tpr,linewidth=2, label=name + ", auc="+str(auc))
    
    plt.legend(loc=4)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()


# 1. Logical Regression Classifier

# In[85]:


from sklearn.linear_model import LogisticRegression

LRmodel=[]

LRmodel.append(('LRImbalanced',LogisticRegression(solver='saga',multi_class='multinomial'),X_train, y_train, X_test, y_test))
performance(LRmodel)


# 2. Random forest classifier

# In[88]:


get_ipython().system('pip install sklearn')


# In[89]:


from sklearn.ensemble import RandomForestClassifier

RFmodel = []

RFmodel.append(('RF IMABALANCED', RandomForestClassifier(),X_train,y_train,X_test,y_test))

performance(RFmodel)


# 3. Gaussian Naïve Bayes Classifier
# 

# In[ ]:




