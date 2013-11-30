# permute_data.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



# Generate a random permutation of the training/validation/test sets; useful for running the minicontest.

import minicontest
import samples
import sys
import util

from dataClassifier import DIGIT_DATUM_HEIGHT,DIGIT_DATUM_WIDTH,contestFeatureExtractorDigit

import numpy as np

def writeLabeledData(prefix, labeled_data):
    datums, labels = zip(*labeled_data)

    with open(prefix + "images", 'w') as f:
        for datum in datums:
            f.write(str(datum) + "\n")
        f.close()

    with open(prefix + "labels", 'w') as f:
        for label in labels:
            f.write(str(label) + "\n")
        f.close()

rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)
rawValidationData = samples.loadDataFile("digitdata/validationimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
validationLabels = samples.loadLabelsFile("digitdata/validationlabels", 1000)
rawTestData = samples.loadDataFile("digitdata/testimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
testLabels = samples.loadLabelsFile("digitdata/testlabels", 1000)


all_data = rawTrainingData + rawValidationData + rawTestData
all_labels = trainingLabels + validationLabels + testLabels

labeled_data = zip(all_data, all_labels)

perm = np.random.permutation(len(labeled_data))

permuted_data = []
for i in perm:
    permuted_data.append(labeled_data[i])
labeled_data = permuted_data

TRAIN_SIZE = 5000
VALIDATE_SIZE = 1000
TEST_SIZE = 1000

new_train_data = labeled_data[0:TRAIN_SIZE]
writeLabeledData("digitdata/contest_train", new_train_data)
new_validate_data = labeled_data[TRAIN_SIZE:TRAIN_SIZE+VALIDATE_SIZE]
writeLabeledData("digitdata/contest_validate", new_validate_data)
new_test_data = labeled_data[TRAIN_SIZE+VALIDATE_SIZE:TRAIN_SIZE+VALIDATE_SIZE+TEST_SIZE]
writeLabeledData("digitdata/contest_test", new_test_data)
