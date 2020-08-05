
# coding: utf-8

# In[35]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Matplotlib is a plotting library for the Python programming language
from sklearn.model_selection import train_test_split #For Spliting Dataset
#The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
#In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
from sklearn.metrics import accuracy_score


# In[36]:



#importing the dataset
data = pd.read_csv('Update (1).csv')

#Before making anything like feature selection,feature extraction and classification, firstly we start with basic data analysis. Lets look at features of data.
data.head()  # head method show only first 5 rows

#. Pandas has a helpful select_dtypes function which we can use to build a new dataframe containing only the object columns.
obj_data = data.select_dtypes(include=['object']).copy()
obj_data.head()   # head method show only first 5 rows

#Input   [Encoding Categorical Values]
X = data.iloc[:,0:18].values 
print("\nX before making numerical: \n",X)


#taking careof categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

X[:,2]= labelencoder_X.fit_transform(X[:,2])
X[:,3]= labelencoder_X.fit_transform(X[:,3])
X[:,5]= labelencoder_X.fit_transform(X[:,5])
X[:,7]= labelencoder_X.fit_transform(X[:,7])
X[:,8]= labelencoder_X.fit_transform(X[:,8])
X[:,12]= labelencoder_X.fit_transform(X[:,12])
X[:,15]= labelencoder_X.fit_transform(X[:,15])
#Print all the numerical data
print("\nX after making numerical: \n",X,"\n")
#After convert all the data in numerical format the attribute name again needed to given.
df = pd.DataFrame(X, columns = ['id', 'diagnosis', 'Invoice Date', 'Date of birth', 'Invoice No', 'Gender', 'Test Name', 'Age',	'Delivery Date', 'Department', 'Sample', 'Contact number', 'patient name', 'Unit', 'Reference Value', 'Address', 'Test Attribute', 'Result'])


# In[37]:


# 1) There is an id that cannot be used for classificaiton 2) Diagnosis is our class label
#Therefore, drop these unnecessary features.
# feature names as a list
col = df.columns       # .columns gives columns names in data 
print(col)

# y includes our labels and x includes our features
y = df.diagnosis
list = ['id','diagnosis']
x = df.drop(list,axis = 1 )
x.head()


# In[38]:


#1) Feature selection with correlation and random forest classification
#Drop Each value using Domain Knowledge.
drop_list1 = ['Invoice No','Invoice Date','Test Name','Delivery Date','Department','Unit','Reference Value','Test Attribute']
x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 
#Create a empty list for storing 4 algorithm output.
p = []
#Store the Feature selection with correlation and random forest classification algorithm output in the empty list.
p.append(x_1.columns.tolist())
#head method show only first 5 rows
x_1.head()

#y includes our labels and x includes our features
#Store the y value in data_dia Variable.
data_dia = y
#Store the x value in data Variable.
data = x


# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators
clf_rf = RandomForestClassifier(random_state=43)      
#fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a .predict() method call.
clr_rf = clf_rf.fit(x_train,y_train)


# In[39]:


# 2) Univariate feature selection and random forest classification

#import SelectKBest For Selecting the top k features that have maximum relevance with the targe variable.
from sklearn.feature_selection import SelectKBest
#import chi2 for scoring the feature
from sklearn.feature_selection import chi2
# find best scored features
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)
#Print all the selected feature score
print('Score list:', select_feature.scores_)
#Print all the selected feature value
print('Feature list:', x_train.columns)
#Store the Univariate feature selection and random forest classification algorithm output in the list.
p.append(x_train.columns.values.tolist())

scores = {}
#For showing selected feature and there output 
#u gets select feature scores and v gets select feature values
for u, v in zip(select_feature.scores_, x_train.columns.values.tolist()):
#Store select feature scores with select feature values
    scores[v] = u
#Print both, select feature scores and select feature values
scores


# In[40]:


#3) Recursive feature elimination (RFE) with random forest
#Create counter1 variable and store x columns by dict
counter1 = dict([(key, 0) for key in x.columns])
#import RFE for selected the feature
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
#create a for loop to run the output and store 100 times
for i in range(100):
    #random forest classifier
    clf_rf_1 = RandomForestClassifier()      
    #recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features.That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
    #estimator = object, step = int or float, optional(default = 1)
    rfe = RFE(estimator=clf_rf_1, n_features_to_select=5, step=1)
    #fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a .predict() method call.
    rfe = rfe.fit(x_train, y_train)
    #Create selected_feature variable and store x_train listed value
    selected_feature = x_train.columns[rfe.support_].values.tolist()
    
    #Create a for loop and run selected_feature values times.
    for f in selected_feature:
        #count how many time each attribute generate output
        counter1[f] = counter1[f] + 1

#Sorted the output in ascending order
s1 = sorted(counter1.items(), key=lambda kv: kv[1], reverse = True)
#Print the sorted result
print(s1)
#Store Recursive feature elimination (RFE) with random forest algorithm output in the list
p.append([x[0] for x in s1[:5]])


# In[41]:


#4) Recursive feature elimination with cross validation and random forest classification

#Now we will not only find best features but we also find how many features do we need.
from sklearn.feature_selection import RFECV
#Create counter2 variable and store x columns by dict
counter2 = dict([(key, 0) for key in x.columns])
#create a for loop to run the output and store 100 times
for i in range(100):
    #random forest classifier
    clf_rf_2 = RandomForestClassifier() 
    #estimator = object, step = int or float, optional(default = 1), cv = int, cross-validation or an iterable optional
    #scoring = string, callable or none, optional, default = none 
    rfecv = RFECV(estimator=clf_rf_2, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    #fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a .predict() method call.
    rfecv = rfecv.fit(x_train, y_train)
    #Create selected_feature variable and store x_train listed value
    selected_feature = x_train.columns[rfecv.support_].values.tolist()
    #Create a for loop and run selected_feature values times.
    for f in selected_feature:
        #count how many time each attribute generate output
        counter2[f] = counter2[f] + 1

#Sorted the output in descending order
s2 = sorted(counter2.items(), key=lambda kv: kv[1], reverse = True)
#Print the sorted result
print(s2)
#Store Recursive feature elimination with cross validation and random forest classification algorithm output in the list
p.append([x[0] for x in s2[:5]])


# In[42]:


#Create a Variable "result" and store the algorithm result 
result = set(p[0])
#Create a for loop and find out the common feature from all algorithm
for s in p[1:]:
    #Update the result depend on common feature
    result.intersection_update(s)
print("Effective Feature: ",result)


# In[43]:


#Create a empty list "output"
output = []
#Create a for loop to showing the result with feature score.
for s in result:
    #Store the result in the empty list
    output.append((s, scores[s]))
#Sorted the output in descending order
output = sorted(output, reverse=True, key=lambda tup: tup[1])
#Showing the result
output


# In[44]:


#Create a for loop for showing the output by numbering
for i, v in enumerate(output):
    #Showing the result by numbering
    print(str(i + 1) + ' :' + v[0])

