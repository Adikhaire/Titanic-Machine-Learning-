import numpy as np
import pandas as pd
from sklearn import svm,metrics
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_true = pd.read_csv('gender_submission.csv')
#print(data.columns.values)
#print (data.isnull().sum())
#print (test.isnull().sum())
target = data['Survived']
data.drop(['PassengerId','Survived','Cabin','Name','Ticket','Embarked','Age','Fare'],axis = 1,inplace = True)
data['Sex'].replace(to_replace = 'male',value = 1,inplace=True)
data['Sex'].replace(to_replace = 'female',value = 0,inplace=True)
#data['Age'].replace(to_replace = np.NaN,value = 30,inplace = True)
test.drop(['PassengerId','Cabin','Name','Ticket','Embarked','Age','Fare'],axis = 1,inplace = True)
test['Sex'].replace(to_replace = 'male',value = 1,inplace=True)
test['Sex'].replace(to_replace = 'female',value = 0,inplace=True)
#test['Age'].replace(to_replace = np.NaN,value = 30,inplace = True)
#test['Fare'].replace(to_replace = np.NaN,value = 30,inplace = True)
clf = svm.SVC(kernel='linear',C =0.1)
clf.fit(data,target)
test_value_predict = clf.predict(test)
test_value = pd.DataFrame(test_value_predict)
print(metrics.log_loss(y_true=y_true['Survived'],y_pred=test_value_predict))
test_value.columns = ['Survived']
test_value.insert(0,'PassengerId',pd.Series(np.arange(892,1310,1),index= test_value.index))
test_value.to_csv(path_or_buf='values.csv',index=False)