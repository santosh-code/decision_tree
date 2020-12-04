import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
data = pd.read_csv("C:/Users/USER/Desktop/DT/Diabetes.csv")
data .columns = [c.replace(' ', '_') for c in data .columns]
dum1=pd.get_dummies(data['_Class_variable'],drop_first=True)


final=pd.concat([data,dum1],axis="columns")
final1=final.drop(['_Class_variable'],axis="columns")

x=final1.iloc[:,:-1]
y=final1.iloc[:,8]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)

clf.fit(train_x,train_y)
y_pred=clf.predict(test_x)
acc=np.mean(test_y==y_pred)
acc#####test acc=0.7

clf.fit(train_x,train_y)
y_pred=clf.predict(train_x)
acc=np.mean(train_y==y_pred)
acc#####train_acc=1

