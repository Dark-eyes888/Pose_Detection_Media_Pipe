# import libraries
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

def file(path):
    df=pd.read_csv(path)
    return df

def label_encode(y):
    le=LabelEncoder().fit_transform(y)
    return le

def Split(data,label,test_size,random_state):
    x=data.drop(columns='class')
    y=label

    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=test_size,random_state=random_state)
    return x_train,x_test,y_train,y_test


def xgb_model(x_train,x_test,y_train,y_test):
    xgb=XGBClassifier(tree_method='gpu_hist').fit(x_train,y_train)
    score=xgb.score(x_train,y_train)
    pred=xgb.predict(x_test)
    acc=accuracy_score(y_test,pred)
    print(f'score:{score},  predictions:{pred},  accuracy:{acc}')
    with open('xgb.pkl','wb') as f:
        pickle.dump(xgb,f)

if __name__=='__main__':
    df=file(r'C:\Users\adm88\ML_deployment\Object_detection\yoga_pose_media_pipe\landmarks.csv')
    le=label_encode(df['class'])
    x_train,x_test,y_train,y_test=Split(df,le,0.2,123)
    xgb_model(x_train,x_test,y_train,y_test)
    
