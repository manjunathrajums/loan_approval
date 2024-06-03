import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
import pickle

data_frame = pd.read_csv('./loan.csv')

data_frame.drop(['loan_id'],axis=1,inplace=True)

data_frame

encoder = LabelEncoder()
scaler = StandardScaler()
data_frame[' self_employed'] = encoder.fit_transform(data_frame[' self_employed'].str.lower())
data_frame[' education'] = encoder.fit_transform(data_frame[' education'].str.lower())
data_frame[' education']=data_frame[' education'].map({0:1,1:0})
data_frame[' loan_status'] = encoder.fit_transform(data_frame[' loan_status'].str.lower())
data_frame[' loan_status'] = data_frame[' loan_status'].map({0:1,1:0})
data_frame.columns=['no_of_dependents', 'education', 'self_employed',
       'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
       'residential_assets_value', 'commercial_assets_value',
       ' luxury_assets_value', 'bank_asset_value', 'loan_status']

data_frame

data_frame['debt_to_income'] = data_frame['income_annum']/data_frame['loan_amount']
data_frame['income_ratio'] = (data_frame['income_annum']*data_frame['loan_term'])/data_frame['loan_amount']

x = data_frame.drop(['loan_status'],axis=1)
x

print(data_frame.debt_to_income.values.tolist())

y = data_frame['loan_status'].values
y

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=41)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled =scaler.transform(x_test)

xgboost_model = XGBClassifier(random_state=41)
xgboost_model.fit(x_train_scaled, y_train)
#print(f"Fitting score of the model is {svm_model.score(x_train_scaled,y)}")
#predicting from the model

y_pred = xgboost_model.predict(x_test_scaled)

accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy score is {accuracy*100:.3f}%")

score= f1_score(y_test,y_pred)
print(f'f1 score is {score}')

pipeline = {
    'xgboost_model':xgboost_model,
    'label_encoder':encoder,
    'standardscaler':scaler
}

with open("model.pkl",'wb') as file:
    pickle.dump(pipeline,file)