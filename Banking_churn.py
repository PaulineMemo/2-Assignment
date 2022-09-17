import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded",
                   page_title='Banking-Churn-Model App')

data = pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
data.head()

data.describe(include='all')

data.info()

#data.shape

#print("Number of rows", data.shape[0])
#print("Number of columns", data.shape[1])

data.isnull().sum()
#data.columns

data=data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

data.head()

data=pd.get_dummies(data,drop_first=True)
data.head()
data['Exited'].value_counts()

X=data.drop('Exited', axis=1)
y=data['Exited']

## Handling imbalanced data

from imblearn.over_sampling import SMOTE
X_res,y_res=SMOTE().fit_resample(X,y)

X_res.value_counts()
y_res.value_counts()

## Splitting Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res, y_res, test_size=0.2, random_state=42)

## Feature scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#X_train


##Training the model

##Logistic Regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train, y_train)

y_pred1=lr.predict(X_test)

from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
accuracy_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
## SVC
from sklearn import svm
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred2 = svm.predict(X_test)
accuracy_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
f1_score(y_test,y_pred2)

##KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred3=knn.predict(X_test)
accuracy_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
f1_score(y_test,y_pred3)

##DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred4=dt.predict(X_test)
accuracy_score(y_test,y_pred4)
precision_score(y_test,y_pred4)
recall_score(y_test,y_pred4)
f1_score(y_test,y_pred4)

##RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5=rf.predict(X_test)
accuracy_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
f1_score(y_test,y_pred5)

## Gradient Boosting Gradient

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred6=gbc.predict(X_test)
accuracy_score(y_test,y_pred6)
precision_score(y_test,y_pred6)
recall_score(y_test,y_pred6)
f1_score(y_test,y_pred6)

final_data= pd.DataFrame({'Models':["LR","SVC","KNN","DT","RF","GBC"],
                         "ACC":[accuracy_score(y_test,y_pred1),
                               accuracy_score(y_test,y_pred2),
                               accuracy_score(y_test,y_pred3),
                               accuracy_score(y_test,y_pred4),
                               accuracy_score(y_test,y_pred5),
                               accuracy_score(y_test,y_pred6)]})
#st.write('final_data')

## Saving the model

X_res=sc.fit_transform(X_res)
model = rf.fit(X_res,y_res)

# saving the model
import pickle
pickle_out = open("classifier.pkl", mode = "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
model = pickle.load(open('classifier.pkl', 'rb'))
html_temp = """
<div style ="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Bank Churn Prediction ML App</h2>
</div>
"""
st.subheader('The app shows whether a customer will churn or not.')
st.markdown(html_temp, unsafe_allow_html=True)

Age = st.number_input("Age")
Tenure = st.number_input("Tenure", )
Balance = st.number_input("Balance")
HasCrCard = st.selectbox("Has Credit Card", options=['No','Yes'])
EstimatedSalary = st.number_input("Estimated Salary")
NumOfProducts = st.number_input("Number of Products")
IsActiveMember = st.selectbox("Is an active member", options=['No','Yes'])
CreditScore = st.number_input("Credit Score")
Gender = st.selectbox("Gender", options=['Male', 'Female'])
Geography = st.selectbox("Geography", options=['Germany', 'France', 'Spain'])


safe_html = """
<div style ="background-color:#F4D03F;padding:10px>
<h2 style="color:white; text-align:center;">Customer will not exit</h2>
</div>
"""
danger_html = """
<div style ="background-color:#F4D03F;padding:10px>
<h2 style="color:white; text-align:center;">Customer will exit</h2>
</div>
"""


def predict_cust(Age, Tenure, Balance, HasCrCard, EstimatedSalary, NumOfProducts, IsActiveMember, CreditScore,Gender, Geography):
      input = np.array([[Age, Tenure, Balance, HasCrCard, EstimatedSalary, NumOfProducts, IsActiveMember, CreditScore]]).astype(np.float64)
      input = np.array([[Gender, Geography,IsActiveMember]]).astype(np.object)
      prediction = model.predict_proba(input)
      pred = np.argmax(prediction)
      return pred


if st.button("Predict"):
      output = predict_cust(Age, Tenure, Balance, HasCrCard, EstimatedSalary, NumOfProducts, IsActiveMember, CreditScore, Gender, Geography)
      st.success("The verdict{}".format(output))

      if output == 0:
            st.markdown(safe_html, unsafe_allow_html=True)
      else:
            st.markdown(danger_html, unsafe_allow_html=True)
