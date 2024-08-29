import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('nearest-earth-objects(1910-2024).csv')

# print(data.head(10))
# print(data.info())

# print(data.isna().sum())
data = data.dropna()


encode = LabelEncoder()
data['is_hazardous'] = encode.fit_transform(data['is_hazardous'])
data['orbiting_body'] = encode.fit_transform(data['orbiting_body'])


# print(data.info())
X = data[['absolute_magnitude','orbiting_body','estimated_diameter_min','estimated_diameter_max','relative_velocity','miss_distance']].values
y = data['is_hazardous'].values

scale = preprocessing.StandardScaler()
nr = scale.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# print('train',X_train.shape,y_train.shape)
# print('test',X_test.shape,y_test.shape)
#creat model #
k = 2

model =KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print(pred)
print('-----------------------------------')
print(y)

acc = accuracy_score(pred,y_test)
print(acc)