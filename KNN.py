import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("Maternal Health Risk Data Set.csv")
X = pd.read_csv("Maternal Health Risk Data Set.csv", usecols=[i for i in data.columns if i != 'RiskLevel' ])
y = data['RiskLevel'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
