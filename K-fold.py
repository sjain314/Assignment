import numpy as np
import pandas as pd
from random import randrange
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def printMetrics(actual, predictions):
    assert len(actual) == len(predictions)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


class kFoldCV:

    def __init__(self):
        pass

    def crossValSplit(self, dataset, numFolds):
        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit

    def kFCVEvaluate(self, dataset, numFolds):
        print('*' * 20)
        print("Using KNN Classifier")
        knn = KNeighborsClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = knn.predict(testSet)

            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print('Scores: %s' % scores)
        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print('*' * 20)
        print('\n')

        print('*' * 20)
        print("Using Decision Tree Classifier")
        DT = DecisionTreeClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            DT.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = DT.predict(testSet)

            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print('Scores: %s' % scores)

        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print('*' * 20)
        print('\n')

        print('*' * 20)
        print("Using Random Forest Classifier")
        RF = RandomForestClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            RF.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = RF.predict(testSet)

            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print('Scores: %s' % scores)

        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print('*' * 20)
        print('\n')

        print('*' * 20)
        print("Using Naive Bayes Classifier")
        NB = GaussianNB()
        folds = self.crossValSplit(dataset, numFolds)
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            NB.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = NB.predict(testSet)

            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print('Scores: %s' % scores)

        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        print('*' * 20)
        print('\n')


def readData(fileName):
    data = []
    labels = []

    with open(fileName, "r") as file:
        lines = file.readlines()
    for line in lines:
        splitline = line.strip().split(',')
        data.append(splitline)
        labels.append(splitline[-1])
    return data, labels


File = 'Maternal Health Risk Data Set.csv'

Data, Label = readData(File)
df = pd.DataFrame(Data)
df = df.apply(preprocessing.LabelEncoder().fit_transform)
Features = df.values.tolist()
Labels = [car[-1] for car in Features]
kfcv = kFoldCV()
print('*' * 20)
print('Maternal Health Risk Data Set')
print('\n')

kfcv.kFCVEvaluate(Features, 10)
