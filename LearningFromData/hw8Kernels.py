from sklearn import svm
import numpy as np
from typing import Union, List, Tuple
import random

# see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

class SVMclassifier:
    """make SVM classifier object. leverages sklearn SVC object. Helper class to make working with SVM simpler"""
    def __init__(self, X, Y, C=.01, Q=2, kernel='poly'):
        self._hypothesis = svm.SVC(C=C, kernel=kernel, degree=Q, gamma=1, coef0=1)
        self._hypothesis.fit(X, Y)

    def test(self, testData):
        """returns the predictions of this classifier on the input TESTDATA"""
        predictions = self._hypothesis.predict(testData)
        return predictions

    def getError(self, testData, testTarget):
        """returns the error % of this classifier"""
        accuracy = self._hypothesis.score(testData, testTarget)
        return 1 - accuracy

    def getNumSupportVectors(self):
        return self._hypothesis.n_support_

    def kFoldValidation(self, k:int, X, Y):
        """runs k fold validation and returns the average of score() - ie the average of the validation errors
        doesn't shuffle the lists X or Y"""
        assert len(X) > k
        lenX = len(X)
        foldSize = lenX // k
        inputFolds = []
        targetFolds = []
        startIndex = 0
        for i in range(k):
            # note: do list concatenation instead of append. As append will wrap each one in its own list and I'll get
            # something like [[array(), array()]]. Which is an unnecessary layer and will mess up SVM fit(). This is
            # true because lst slicing returns a list. So if I append this list into an existing list it will be a
            # list of lists vs. just a list of arrays (which are the input vectors) that I want
            # inputFolds += X[startIndex:startIndex+foldSize]
            # targetFolds += Y[startIndex:startIndex+foldSize]
            inputFolds.append(X[startIndex:startIndex+foldSize])
            targetFolds.append(Y[startIndex:startIndex+foldSize])
            startIndex += foldSize
        for index, remainderIndex in enumerate(range(startIndex, lenX)):
            inputFolds[index].append(X[remainderIndex])
            targetFolds[index].append(Y[remainderIndex])
        errors = []
        for index in range(k):
            trainX = inputFolds[0:index] + inputFolds[index+1:]
            # flatten list by one level. (below). Therefore instead of list of lists we want a list of np arrays
            # (ie the input vectors). We do the same for the target
            trainX = [element for subList in trainX for element in subList]
            trainY = targetFolds[0:index] + targetFolds[index+1:]
            trainY = [element for subList in trainY for element in subList]
            validX = inputFolds[index]
            validY = targetFolds[index]
            # validY = validY.reshape(-1, 1)
            self._hypothesis.fit(trainX, trainY)
            error = 1 - self._hypothesis.score(validX, validY)
            errors.append(error)
        return np.mean(errors)

def oneVersusMany(targets: Union[Tuple, List], one: Union[str, int, float]):
    """helper function that changes the labels to simple +1/-1 labels. +1 for the value of ONE and -1 for values
    that do not equal ONE"""
    newTargets = []
    count = 0
    for target in targets:
        count += 1
        if target == one:
            newTargets.append(1)
        else:
            newTargets.append(-1)
    return newTargets

def oneVersusOne(inputs: Union[Tuple, List], targets: Union[Tuple, List], one: Union[str, int, float],
                 other: Union[str, int, float]):
    """helper function that labels +1 for ONE and -1 for the OTHER. For all other is ignores. therefore the output
    list will most likely be smaller than input TARGETS. Also, adjusts the training input to throw out the
    corresponding inputs that map to the ignored values"""
    newTargets = []
    newInputs = []
    for index, target in enumerate(targets):
        if one == target:
            newTargets.append(1)
            newInputs.append(inputs[index])
        elif other == target:
            newTargets.append(-1)
            newInputs.append(inputs[index])
    return newInputs, newTargets

def getRawData():
    """helper function that gets the data from the .dta files. Returns 4 arrays (1) input data for the test set
    (2) target values for test set (3) input data for training set (4) target values for training set"""
    rawTestData = np.loadtxt('/home/harrison/LearningFromDataCode/hw8FeaturesTest.dta')
    testInput = rawTestData[:, [1, 2]]
    testTarget = rawTestData[:, [0]]
    rawTrainData = np.loadtxt('/home/harrison/LearningFromDataCode/hw8FeaturesTrain.dta')
    trainInput = rawTrainData[:, [1, 2]]
    trainTarget = rawTrainData[:, [0]]
    return testInput, testTarget, trainInput, trainTarget

testInput, testTargetRaw, trainInput, trainTargetRaw = getRawData()
# changing the labels to still fit the SVM binary classification defaults but now just label all positive labels as the
# number we want to label and all other labels as negative labels
zeroTarget = oneVersusMany(trainTargetRaw, 0)
twoTarget = oneVersusMany(trainTargetRaw, 2)
fourTarget = oneVersusMany(trainTargetRaw, 4)
sixTarget = oneVersusMany(trainTargetRaw, 6)
eightTarget = oneVersusMany(trainTargetRaw, 8)

# below prints out the possible choices to problem 2. See that the zeroSVM error is the highest
zeroSVM = SVMclassifier(trainInput, zeroTarget)
twoSVM = SVMclassifier(trainInput, twoTarget)
fourSVM = SVMclassifier(trainInput, fourTarget)
sixSVM = SVMclassifier(trainInput, sixTarget)
eightSVM = SVMclassifier(trainInput, eightTarget)
print("problem 2 in sample errors")
print(zeroSVM.getError(trainInput, zeroTarget))
print(twoSVM.getError(trainInput, twoTarget))
print(fourSVM.getError(trainInput, fourTarget))
print(sixSVM.getError(trainInput, sixTarget))
print(eightSVM.getError(trainInput, eightTarget))

print("\nproblem 3 in sample errors")
oneTarget = oneVersusMany(trainTargetRaw, 1)
threeTarget = oneVersusMany(trainTargetRaw, 3)
fiveTarget = oneVersusMany(trainTargetRaw, 5)
sevenTarget = oneVersusMany(trainTargetRaw, 7)
nineTarget = oneVersusMany(trainTargetRaw, 9)
oneSVM = SVMclassifier(trainInput, oneTarget)
threeSVM = SVMclassifier(trainInput, threeTarget)
fiveSVM = SVMclassifier(trainInput, fiveTarget)
sevenSVM = SVMclassifier(trainInput, sevenTarget)
nineSVM = SVMclassifier(trainInput, nineTarget)
print(oneSVM.getError(trainInput, oneTarget))
print(threeSVM.getError(trainInput, threeTarget))
print(fiveSVM.getError(trainInput, fiveTarget))
print(sevenSVM.getError(trainInput, sevenTarget))
print(nineSVM.getError(trainInput, nineTarget))

print("problem 4 # of support vectors")
# note: to get the answer for problem 4. you take the sum of the 2 elements within each array and get the different
# of that. Each number is the number of support vectors for each class (ie only 2 classes because we are doing binary)
print(zeroSVM.getNumSupportVectors(), oneSVM.getNumSupportVectors())

# problem 5 answer choices printed below
oneVersusFiveInput, oneVersusFiveTarget = oneVersusOne(trainInput, trainTargetRaw, 1, 5)
oneVersusFiveTestInput, oneVersusFiveTestTarget = oneVersusOne(testInput, testTargetRaw, 1, 5)
oneVFiveSVM = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.001)
oneVFiveSVMa = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.01)
oneVFiveSVMb = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.1)
oneVFiveSVMc = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=1)
print("number of support vectors")
print(oneVFiveSVM.getNumSupportVectors(), oneVFiveSVMa.getNumSupportVectors(), oneVFiveSVMb.getNumSupportVectors(),
      oneVFiveSVMc.getNumSupportVectors())
print("out of sample errors")
print(oneVFiveSVM.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget),
      oneVFiveSVMa.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget),
      oneVFiveSVMb.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget),
      oneVFiveSVMc.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))
print("in sample errors")
print(oneVFiveSVM.getError(oneVersusFiveInput, oneVersusFiveTarget),
      oneVFiveSVMa.getError(oneVersusFiveInput, oneVersusFiveTarget),
      oneVFiveSVMb.getError(oneVersusFiveInput, oneVersusFiveTarget),
      oneVFiveSVMc.getError(oneVersusFiveInput, oneVersusFiveTarget))
#end of problem 5 anwer choices

# problem 6 answer choices below
oneVFiveSVM = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.0001, Q=2)
oneVFiveSVMa = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.0001, Q=5)
print("in sample error comparison")
print(oneVFiveSVM.getError(oneVersusFiveInput, oneVersusFiveTarget),
      oneVFiveSVMa.getError(oneVersusFiveInput, oneVersusFiveTarget))
oneVFiveSVMb = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.001, Q=2)
oneVFiveSVMc = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.001, Q=5)
print("number of support vectors comparison")
print(oneVFiveSVMb.getNumSupportVectors(),
      oneVFiveSVMc.getNumSupportVectors())
oneVFiveSVMd = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.01, Q=2)
oneVFiveSVMe = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.01, Q=5)
print("in sample error comparison")
print(oneVFiveSVMd.getError(oneVersusFiveInput, oneVersusFiveTarget),
      oneVFiveSVMe.getError(oneVersusFiveInput, oneVersusFiveTarget))
oneVFiveSVMf = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=1, Q=2)
oneVFiveSVMg = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=1, Q=5)
print("out of sample error comparison")
print(oneVFiveSVMf.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget),
      oneVFiveSVMg.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))

# end of problem 6 answer choices
# print problem 7 answer choices
oneVFiveSVMh = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.1, Q=2)
# the 2 lists below correspond to increasing C values from lowest C to highest C
SVMs = [oneVFiveSVM, oneVFiveSVMb, oneVFiveSVMd, oneVFiveSVMh, oneVFiveSVMf]
modelCounts = [0, 0, 0, 0, 0]
lowestCVs = []
# an indices list to be shuffled each of the 100 runs to simulate randomized k-fold CV
indices = [i for i in range(len(oneVersusFiveInput))]
for _ in range(100):
    errors = []
    random.shuffle(indices)
    for index, model in enumerate(SVMs):
        oneVersusFiveInputShuffled = [oneVersusFiveInput[index] for index in indices]
        oneVersusFiveTargetShuffled = [oneVersusFiveTarget[index] for index in indices]
        CVerror = model.kFoldValidation(10, oneVersusFiveInputShuffled, oneVersusFiveTargetShuffled)
        errors.append((CVerror, index))
        # added the below because figured out that the best performing one (ie answer to 7) was when index = 1 ie
        # the model when C=.001
        if index == 1:
            lowestCVs.append(CVerror)
    errors.sort()
    lowestCVerror = errors[0]
    modelCounts[lowestCVerror[1]] += 1

print("The counts of which model won via cross validation error",modelCounts)

#problem 8
print("the value of the lowest CV error average", np.mean(lowestCVs))

#problem 9
# rbf kernel SVMs
oneVFiveSVMrbf = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=.01, kernel='rbf')
oneVFiveSVMrbfa = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=1, kernel='rbf')
oneVFiveSVMrbfb = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=100, kernel='rbf')
oneVFiveSVMrbfc = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=10000, kernel='rbf')
oneVFiveSVMrbfd = SVMclassifier(oneVersusFiveInput, oneVersusFiveTarget, C=1000000, kernel='rbf')

print("in sample errors from lowest value of C to highest")
print(oneVFiveSVMrbf.getError(oneVersusFiveInput, oneVersusFiveTarget))
print(oneVFiveSVMrbfa.getError(oneVersusFiveInput, oneVersusFiveTarget))
print(oneVFiveSVMrbfb.getError(oneVersusFiveInput, oneVersusFiveTarget))
print(oneVFiveSVMrbfc.getError(oneVersusFiveInput, oneVersusFiveTarget))
print(oneVFiveSVMrbfd.getError(oneVersusFiveInput, oneVersusFiveTarget))

print("out of sample errors from lowest value of C to highest")
print(oneVFiveSVMrbf.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))
print(oneVFiveSVMrbfa.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))
print(oneVFiveSVMrbfb.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))
print(oneVFiveSVMrbfc.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))
print(oneVFiveSVMrbfd.getError(oneVersusFiveTestInput, oneVersusFiveTestTarget))