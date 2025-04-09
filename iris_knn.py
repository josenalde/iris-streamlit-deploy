import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle
import numpy as np

df = pd.read_csv('datasets/iris.csv')
db = df.copy()
db.drop(columns=['Id'], axis=1, inplace=True)

encoder = preprocessing.LabelEncoder()
db['Species_encoded'] = encoder.fit_transform(df['Species'])

X = np.array(db.drop(
    columns=['Species', 'Species_encoded', 'SepalLengthCm', 'SepalWidthCm'], axis=1))
y = np.array(db['Species_encoded'])

knn_model = KNeighborsClassifier(n_neighbors=9)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30, stratify=y)

knn_model.fit(X_train, y_train)

s = y_test.size

y_pred = []
for i in range(s):
    y_pred.append(knn_model.predict([X_test[i, :]]))

print(accuracy_score(y_true=y_test, y_pred=y_test))
with open('models/model-knn.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
