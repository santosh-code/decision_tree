import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
data = pd.read_csv("C:/Users/USER/Desktop/DT/Company_Data.csv")
data.head(15)
data.describe()

data.Sales=pd.cut(data.Sales,bins=[-1,10,20],labels=['A','B'])
dum1_sales=pd.get_dummies(data['Sales'],drop_first=True)
dum2_ShelveLoc=pd.get_dummies(data['ShelveLoc'],drop_first=True)
dum3_Urban=pd.get_dummies(data['Urban'],drop_first=True)
dum4_US=pd.get_dummies(data['US'],drop_first=True)
drop=data.drop(['ShelveLoc','Urban','US','Sales'],axis='columns')

final1=pd.concat([drop,dum1_sales,dum2_ShelveLoc,dum4_US,dum3_Urban],join='inner',axis='columns')
final=final1.iloc[:,[7,0,1,2,3,4,5,6,8,9,10,11]]

x=final.iloc[:,1:]
y=final.iloc[:,0]
# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train_x,train_y)

# Prediction on Train Data
preds=model.predict(train_x)
np.mean(preds==train_y) # Train Data Accuracy


# Prediction on Test Data
preds = model.predict(test_x)
np.mean(preds==test_y) # Test Data Accuracy 

##########pre-pruning

params = {'max_depth': [2,4,6,8,10,12],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}
gcv = GridSearchCV(estimator=model,param_grid=params)
gcv.fit(train_x,train_y)

model1 = gcv.best_estimator_
model1.fit(train_x,train_y)
pre=model1.predict(train_x)
np.mean(pre==train_y)##train acc=0.86

pre=model1.predict(test_x)
np.mean(pre==test_y)##test acc=0.83
