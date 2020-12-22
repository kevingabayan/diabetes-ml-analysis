"""
This is a simple ML analysis on a diagonistic diabetes data set.
thanks to user @visabh123 for providing the framework to allow for this analysis.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def predict(algorithm_str, y_test, x_test, to_predict):
    # Predicting the Test set results
    y_pred = to_predict.predict(x_test)

    # Creating the confusion Matrix
    # True Negative = 0, 0
    # False Negative = 1, 0
    # True positive = 1, 1
    # False positive = 0, 1
    cm = confusion_matrix(y_test, y_pred)
    c = cm[0, 0] + cm[1, 1]
    total = 0
    for i in cm:
        total += i.sum()
    accuracy = (c/total) * 100
    print(algorithm_str + str(round(accuracy, 2)) + "%")


# Import dataset
dataset = pd.read_csv('diabetes-ml-data.csv')
# Shuffle dataset because it is sorted by outcome
dataset = dataset.sample(frac=1, random_state=42)
# X = features
X = dataset.iloc[:, 0:8].values
# Y = what we are going to predict
Y = dataset.iloc[:, 8].values

dataset.groupby('Outcome').hist(figsize=(12, 12))

# Splitting the dataset into the Training set and Test set
# Training set = known output, model learns on this data
# Test set = test our model's prediction on this subset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
predict("Logistic Regression: ", Y_test, X_test, classifier)

# Fitting K-Nearest Neighbors to the training set
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)
predict("K-NN: ", Y_test, X_test, classifier)

# Fitting SVM
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, Y_train)
predict("Support Vector Machines: ", Y_test, X_test, classifier)

# Fitting K-SVM
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)
predict("K-SVM: ", Y_test, X_test, classifier)

# Fitting Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
predict("Naive Bayes: ", Y_test, X_test, classifier)

# Fitting Decision Tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
predict("Decision Tree: ", Y_test, X_test, classifier)

# Fitting Random Forest
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
predict("Random Forest: ", Y_test, X_test, classifier)

