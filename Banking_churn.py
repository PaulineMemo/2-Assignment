import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

import numpy as np

data=pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
st.write(data.head())
st.write(data.describe())
st.write(data.isna().sum())

#Removing columns that have no effect on customer churn
(data.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True))
st.write(data.head())
st.write(data.describe())
st.write(data.groupby('Exited').mean())

st.write("Create one hot encoded columns for Geography and Gender")

# convert categorical columns in text to numerical values i.e Gender and Geography
tempdata=data.drop(['Geography','Gender'], axis=1)
st.write(tempdata)
Geography=pd.get_dummies(data.Geography).iloc[:,1:]
Gender=pd.get_dummies(data.Gender).iloc[:,1:]

data1=pd.concat([tempdata,Geography,Gender], axis=1)
st.write(data1.head())

#separating the features and labels

#splitting the data
x_train = data1.drop(['Exited'], axis=1)
y_train = data1['Exited']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=21)

st.write(x_train.head())
st.write(y_train.head())

st.write(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Training and modelling the train data

classifier=rfc(n_estimators=200, random_state=0)
classifier.fit(x_train,y_train)
predicted_labels = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
st.write((classification_report(y_test, predicted_labels)))
st.write((confusion_matrix(y_test, predicted_labels)))
st.write((accuracy_score(y_test, predicted_labels)))
from sklearn.svm import SVC as svc

st.write("Train the model using support vector machines:")
svc_object = svc(kernel='rbf', degree=8)
svc_object.fit(x_train, y_train)
predicted_labels = svc_object.predict(x_test)
st.write((classification_report(y_test, predicted_labels)))
st.write((confusion_matrix(y_test, predicted_labels)))
st.write((accuracy_score(y_test, predicted_labels)))

st.write("Train the model using logistic regression:")
from sklearn.linear_model import LogisticRegression
lr_object = LogisticRegression()
lr_object.fit(x_train, y_train)
predicted_labels = lr_object.predict(x_test)
st.write((classification_report(y_test, predicted_labels)))
st.write((confusion_matrix(y_test, predicted_labels)))
st.write((accuracy_score(y_test, predicted_labels)))


st.sidebar.title("Customer Banking Churn App")

import pickle
st.text_input("Age")
st.text_input("CreditScore")
st.text_input("NumOfProducts")
st.text_input("Geography")
st.text_input("Gender")
st.text_input("IsActiveMember")
st.text_input("EStimatedSalary")
st.text_input("Tenure")
