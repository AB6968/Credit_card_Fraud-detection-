#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


df=pd.read_csv("creditcard.csv")


# In[4]:


df


# # Exploratory data analysis

# In[5]:


df.shape


# In[16]:


df.drop_duplicates(subset=None, inplace=True)
df.shape


# In[6]:


df.info()

# no missing values


# In[7]:


df.describe()


# In[19]:


plt.figure(figsize=(8,8))
plt.title('Transaction Time Distributions')
sns.distplot(df['Time'])
plt.show()


# In[8]:


df['Class'].unique()


# In[9]:


df['Class'].value_counts()
# data is imbalanced


# Data visualization

# In[10]:


import matplotlib.pyplot as plt


# In[10]:


df.hist(figsize=[30,35])
plt.show()


# In[11]:


df.hist("Class")


# In[12]:


df.boxplot(figsize=(25,7),grid=False)


# In[14]:


sns.boxplot(df['Amount'])


# In[15]:


# sns.pairplot(df)


# In[11]:


# Here 'amount' and other feature variable ranges are different so we need to standardize the 'amount'
# Scale amount by Standardization
from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
df['amount_scaled'] = ss.fit_transform(df['Amount'].values.reshape(-1,1))


# In[12]:


df


# In[18]:


sns.boxplot(df['amount_scaled'])


# In[19]:


df.boxplot(figsize=(25,7),grid=False)


# In[20]:


df


# In[21]:


# split into feature variable
x=df.drop(["Amount", "Class"], axis=1)
x


# In[22]:


# split into response variable
y = df["Class"]
y


# # Model Fitting

# In[23]:


#!pip install imblearn


# In[24]:


# import libraries 
# splitting data 
from sklearn.model_selection import train_test_split

# to check performance of metrics
from sklearn import metrics


# In[25]:


# balancing data
from imblearn.under_sampling import RandomUnderSampler


# In[26]:


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3, shuffle=True)

print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_test: ",X_test.shape)
print("y_test: ",y_test.shape)

print('\n')
print('............')
print('\n')


# data is imbalanced hence we use under sampling techniques

rus= RandomUnderSampler(sampling_strategy='majority')
X_train_under,y_train_under=rus.fit_resample(X_train, y_train)
X_test_under, y_test_under = X_test, y_test

print("X_train_under: ",X_train_under.shape)
print("y_train_under: ",y_train_under.shape)
print("X_test_under: ",X_test_under.shape)
print("y_test_under: ",y_test_under.shape)


# In[27]:


#!pip3 install -U scikit-learn


# In[28]:


# split training and test data
#X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3, shuffle=True)


# # Classification Algorithms

# In[29]:


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

# In[30]:


from sklearn.linear_model import LogisticRegression

LRmodel=[]

LRmodel.append(('LRImbalanced',LogisticRegression(solver='saga',multi_class='multinomial'),X_train, y_train, X_test, y_test))
LRmodel.append(('LRImbalanced',LogisticRegression(solver='saga',multi_class='multinomial'),X_train_under, y_train_under, X_test_under, y_test_under))
performance(LRmodel)


# 2. Random forest classifier

# In[31]:


#!pip install sklearn


# In[33]:


from sklearn.ensemble import RandomForestClassifier

RFmodel = []

RFmodel.append(('RF IMABALANCED', RandomForestClassifier(),X_train,y_train,X_test,y_test))
RFmodel.append(('RF Undersample', RandomForestClassifier(), X_train_under, y_train_under, X_test_under, y_test_under))

performance(RFmodel)


# 3. Gaussian Naïve Bayes Classifier
# 

# In[34]:


from sklearn.naive_bayes import GaussianNB

NBmodel = []

NBmodel.append(("NB Imbalanced", GaussianNB(), X_train, y_train, X_test, y_test))
NBmodel.append(('NB Undersample', GaussianNB(), X_train_under, y_train_under, X_test_under, y_test_under))

performance(NBmodel)


# 4. Decision tree classifier

# In[35]:


from sklearn.tree import DecisionTreeClassifier

DTmodel = []

DTmodel.append(("DT Imbalanced", DecisionTreeClassifier(), X_train, y_train, X_test, y_test))
DTmodel.append(('DT Undersample', DecisionTreeClassifier(), X_train_under, y_train_under, X_test_under, y_test_under))

performance(DTmodel)


# 5. K-Nearest Neighbours

# In[41]:


from sklearn.neighbors import KNeighborsClassifier

KNNmodel = []

KNNmodel.append(("KNN Imbalanced", KNeighborsClassifier(), X_train, y_train, X_test, y_test))
KNNmodel.append(('KNN Undersample', KNeighborsClassifier(), X_train_under, y_train_under, X_test_under, y_test_under))

performance(KNNmodel)


# In[45]:


# !pip install xgboost


# 6. XG Boost classifier

# In[46]:


from xgboost import XGBClassifier

xgBOOST = []

xgBOOST.append(("XGBoost Imbalanced", XGBClassifier(), X_train, y_train, X_test, y_test))
xgBOOST.append(('XGBoost Undersample', XGBClassifier(), X_train_under, y_train_under, X_test_under, y_test_under))

performance(xgBOOST)


# # comparison of different models

# In[47]:


comparision={
    'Model': names,
    'Accuracy': accuracy_tests,
    'AUC': aucs_tests,
    'Precision Score' : precision_tests,
    'Recall Score': recall_tests, 
    'F1 Score': f1_score_tests
}
print("Comparing performance of various Classifiers: \n \n")
comparision=pd.DataFrame(comparision)
comparision.sort_values('F1 Score',ascending=False)


# # Conclusion:
# The F1 score of XGBoost(Imbalanced) is maximum. It is followed by that of Random Forest(Imbalanced) and then DT(Imbalanced).
# Whereas Logistic Regression have zero F1-Score.
