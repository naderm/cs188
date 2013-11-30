# minicontest.py
# --------------
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


import util
import classificationMethod

class contestClassifier(classificationMethod.ClassificationMethod):
    """
    Create any sort of classifier you want. You might copy over one of your
    existing classifiers and improve it.
    """
    def __init__(self, legalLabels):
        self.guess = None
        self.type = "minicontest"

    def train(self, data, labels, validationData, validationLabels):
        """
        Please describe your training procedure here.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, testData):
        """
        Please describe how data is classified here.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
