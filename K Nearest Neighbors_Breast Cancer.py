import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

df = pd.read_csv('../datasets/UWMadisonBreastCancer.txt')
df.replace('?', -112233, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Accuracy: {}'.format(accuracy))

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [7,1,1,5,1,6,2,9,4], [3,8,2,1,1,2,5,2,8]])
example_measures = example_measures.reshape(len(example_measures), -1)
result = clf.predict(example_measures)
print(result)
