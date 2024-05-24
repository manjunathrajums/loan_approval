import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv("loan.csv")
df.head(5)
df.tail(5)
df.describe()
df.info()
df.isnull().sum()
df.drop(['loan_id'],axis=1)
df.nunique()
df[' education'].value_counts()
df[' loan_status'].value_counts()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df[' education'] = label_encoder.fit_transform(df[' education'])
df[' self_employed'] = label_encoder.fit_transform(df[' self_employed'])
df[' loan_status'] = label_encoder.fit_transform(df[' loan_status'])

df.head()
education_counts = df[' education'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', colors=['tomato', 'orange'])
plt.title('Education Distribution')
plt.show()
import seaborn as sns
sns.countplot(x = ' no_of_dependents', data = df).set_title('Number of Dependents')
sns.countplot(data=df, x=" education", hue=" loan_status")
plt.show()
sns.countplot(data=df, x=" self_employed", hue=" loan_status")
plt.show()
x = df.drop(columns = [' loan_status'])
y = df[' loan_status']
x.head()
y.head()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=9)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
scores = []
names = []
names.append("Logistic Regression")
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
score_logreg = accuracy_score(y_test, y_pred_lr)

scores.append(score_logreg)
print("Logistic Regression Accuracy: ", score_logreg)

import pickle
pickle.dump(lr,open('model.pkl'))
