# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()
        self.weights = {}

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        def f_2_2(features):
            return sum([i ** 2 for i in features.values()])

        weights = {}
        def train_c(c):
            print "Testing parameter C:", c

            # Reset the weights
            self.weights = dict((label, util.Counter()) for label in self.legalLabels)

            for iteration in range(self.max_iterations):
                print "Starting MIRA iteration ", iteration, "..."
                for features, y in zip(trainingData, trainingLabels):
                    y_p = self.classify([features])[0]
                    if y != y_p:
                        tau = min([
                            c,
                            ((self.weights[y_p] - self.weights[y]) * features + 1.)
                            / (2 * (features * features))
                            ])
                        delta = features.copy()
                        for key, value in delta.items():
                            delta[key] = value * tau
                        self.weights[y] += delta
                        self.weights[y_p] -= delta

            weights[c] = self.weights
            return sum(int(y == y_p) for y, y_p in zip(validationLabels,
                                                       self.classify(validationData)))

        c_scores = [train_c(c) for c in Cgrid]

        # Pick out the best value for C, choosing the lower value in the case of ties
        max_c, max_c_score = Cgrid[0], -1
        for c, c_score in zip(Cgrid, c_scores):
            if c_score > max_c_score or \
              (c_score == max_c_score and c < max_c):
              max_c, max_c_score = c, c_score

        self.weights = weights[max_c]
        self.C = max_c
        return max_c

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                         w_label1 - w_label2

        """
        diff = self.weights[label1] - self.weights[label2]
        for key, value in diff.items():
            diff[key] = abs(value)
        return diff.sortedKeys()[:100]
