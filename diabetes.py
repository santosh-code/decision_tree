import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
data = pd.read_csv("C:/Users/USER/Desktop/DT/Diabetes.csv")
dum1=pd.get_dummies(data['_Class_variable'],drop_first=True)
data .columns = [c.replace(' ', '_') for c in data .columns]

final=pd.concat([data,dum1],axis="columns")
final1=final.drop(['_Class_variable'],axis="columns")

x=final1.iloc[:,:-1]
y=final1.iloc[:,8]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train_x,train_y)

# Prediction on Train Data
preds=model.predict(train_x)
np.mean(preds==train_y) # Train acc=1.0


# Prediction on Test Data
preds = model.predict(test_x)
np.mean(preds==test_y) # Test Data Accuracy =0.74

##########pre-pruning

params = {'max_depth': [2,4,6,8,10,12],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}
gcv = GridSearchCV(estimator=model,param_grid=params)
gcv.fit(train_x,train_y)

model1 = gcv.best_estimator_
model1.fit(train_x,train_y)
pre=model1.predict(train_x)
np.mean(pre==train_y)##train acc=0.79

pre=model1.predict(test_x)
np.mean(pre==test_y)##test acc=0.71
