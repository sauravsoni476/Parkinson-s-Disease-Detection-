import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

parki= pd.read_csv("parkinsons.csv")

parki.drop('name', axis=1, inplace= True)

x = parki.drop('status', axis=1)
y = parki.status
y = y.apply(lambda x: 'Parkinson Disease' if x== 1 else 'No parkinson Disease')

X_train, X_test, y_train,  y_test = train_test_split(x, y, test_size=0.2, random_state=2)

## Data Standardization 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

## now buildt the model
model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)


#############################################################
pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
#print(model.predict([[2,9,6]]))

#model_pridict= model.predict(X_test)

#actual_pridict = pd.DataFrame({'Actual':y_test, 'Pridict':model_pridict})

#accuracy_score(y_test, model_pridict)



