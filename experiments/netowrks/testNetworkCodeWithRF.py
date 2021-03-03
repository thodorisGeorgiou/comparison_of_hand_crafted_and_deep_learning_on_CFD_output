#Test deep features used on Random forests
import os
import sys
import numpy
import pickle
import sklearn.ensemble
import preprocessing

trainData = numpy.load("trainFC2Codes_2FC.npy")
trainLabels = numpy.load(preprocessing.trainLabelPath).astype(numpy.float32)
testData = numpy.load("testFC2Codes_2FC.npy")
testLabels = numpy.load(preprocessing.testLabelPath).astype(numpy.float32)

print("Training Random Forests")
rf = sklearn.ensemble.RandomForestRegressor(n_jobs=88)
rf.fit(trainData, trainLabels)

print("Predicting")
preds = rf.predict(testData)

diff = numpy.square(preds-testLabels)
mean = numpy.sqrt(numpy.mean(diff, axis=0))

numpy.save("vcRF2Predictions2", preds)
numpy.save("vcRF2Gt2", testLabels)

log = open("vcNet2FC2WithRF.acc", "w")
print("Performance")
print(mean, file=log)

