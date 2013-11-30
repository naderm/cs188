# runMinicontest.py
# -----------------
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


# This file is for running the minicontest submission.

import minicontest
import samples
import sys
import util
import pickle
from dataClassifier import DIGIT_DATUM_HEIGHT,DIGIT_DATUM_WIDTH,contestFeatureExtractorDigit

TEST_SIZE = 1000

MINICONTEST_PATH = "minicontest_output.pickle"


if __name__ == '__main__':
    print "Loading training data"
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", 100,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", 100)
    rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)


    featureFunction = contestFeatureExtractorDigit
    legalLabels = range(10)
    classifier = minicontest.contestClassifier(legalLabels)

    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)

    print "Writing classifier output..."
    outfile = open(MINICONTEST_PATH,'w')
    output = {}
    output['guesses'] = guesses;
    pickle.dump(output,outfile)
    outfile.close()
    print "Write successful."
