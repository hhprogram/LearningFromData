from Homework6Regularization import getRawData, LinearRegression, nonLinearTransform, getTestError, getPredictions
import numpy as np
from random import randint
from LinearRegression import get_target_function

trainingPercent = 25 / 35

def splitTraining(trainingIndex=25, first=True):
    """helper function that takes the HW6 dta data set and then splits training group by TRAININGINDEX. if FIRST then
    gets the first elements up to TRAININGINDEX. If LAST then gets all elements from TRAININGINDEX and to the end.
    returns the same tuple as getRawData but then splits TRAININPUT and TRAINTARGET into train and validation sets"""
    trainInput, trainTarget, testInput, testTarget = getRawData()
    validationGroup = trainInput[trainingIndex:] if first else trainInput[:trainingIndex]
    validationTarget = trainTarget[trainingIndex:] if first else trainTarget[:trainingIndex]
    traingGroup = trainInput[:trainingIndex] if first else trainInput[trainingIndex:]
    trainingTarget = trainTarget[:trainingIndex] if first else trainTarget[trainingIndex:]
    return traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget

def shortenZVector(zData, kValue):
    """helper function that takes the array of z vectors returned by the NonLinear Transform method and shortens
    each z vector to be only up to and including the kValue (ie if kValue = 3 then converts each z vector to be just
    (1, z1, z2, z3)"""
    newZData = []
    for zPoint in zData:
        newZData.append(zPoint[:kValue+1])
    return np.array(newZData)

def validateModels(traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget,
                   kValues=[3,4,5,6,7]):
    """trains (using linear regression) models and then validates them. outputs the list of tuples. KVALUES is a list
    of k values to denote up to what index of the nonLinear transform found in nonLinearTransform method we include
    for the zData. Ie if it is 3 then transform input data to zVector of (1, z1, z2, z3)
    (validation error, test error, transform Index) for problem 1"""
    models = []

    for k in kValues:
        zData = shortenZVector(nonLinearTransform(traingGroup), k)
        zValidation = shortenZVector(nonLinearTransform(validationGroup), k)
        validationPredictions, e_inError, weights = LinearRegression(zData, trainingTarget, zValidation)
        validationError = getTestError(validationPredictions, validationTarget)
        zTestInput = shortenZVector(nonLinearTransform(testInput), k)
        testPredictions = getPredictions(weights, zTestInput)
        testError = getTestError(testPredictions, testTarget)
        models.append((validationError, testError, k))
    return models


traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget = splitTraining()
models = validateModels(traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget)
models.sort()
print(models)
# prints out the answer to question 1 and question 2 (first part of tuple is validation error. 2nd is test error 3rd is k value)
# had problems with this problem because my LinearRegression function in HW6 file was non linear transforming the inputs to
# to z space of k = 7 by default no matter the input so that was messing things up
print(models[0])
# sorting it by the 2nd element of the tuple
models.sort(key=lambda x: x[1])
problem1TestError = models[0][1]

# below prints out answers to problems 3 and 4. Basically, same as 1 and 2 but just trains on last 10 and validates on first 25
traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget = splitTraining(trainingIndex=25, first=False)
models = validateModels(traingGroup, trainingTarget, validationGroup, validationTarget, testInput, testTarget)
models.sort()
print(models)
# sorting it by the 2nd element of the tuple
models.sort(key=lambda x: x[1])
problem3TestError = models[0][1]

def problem5(pts, ptTwo):
    """"""
    distances = []
    for pt in pts:
        distance = np.sqrt((pt[0] - ptTwo[0])**2 + (pt[1] - ptTwo[1])**2)
        distances.append((distance, pt))
    return distances

# gets all the euclidean distances from the given points to the problem1 and 3 test errors. We basically make
# problem1 test error the x value and problem3 test error the y value. And then get euclidean distances between that
# point and the given (x, y) point
problem5pts = [(0.0, 0.1), (0.1, 0.2), (0.1, 0.3), (0.2, 0.2), (0.2, 0.3)]
distances = problem5(problem5pts, (problem1TestError, problem3TestError))
distances.sort()
print(distances)

def getAllbut(lst, index):
    """helper function that returns a new list that gets all elements except for the element at INDEX"""
    newLst = []
    for count, element in enumerate(lst):
        if index != count:
            newLst.append(element)
    return newLst


def problem7():
    """problem 7. Which test cross validation"""
    possibleX = [np.sqrt(np.sqrt(3) + 4), np.sqrt(np.sqrt(3) - 1), np.sqrt(9 + 4*np.sqrt(6)), np.sqrt(9 - np.sqrt(6))]
    ro = -4
    constantErrors = []
    linearErrors = []
    for roIndex, x in enumerate(possibleX):
        ro = x
        dataPoints = [(-1, 0), (ro, 1), (1, 0)]
        constantRoError = []
        linearRoError = []
        for count, pt in enumerate(dataPoints):
            trainingData = getAllbut(dataPoints, count)
            validationData = dataPoints[count]
            print(validationData)
            # constant model is the the average 'y' value of the 2 training points
            constantWeight = np.mean([value[1] for value in trainingData])
            # basically just 'learn' the linear model by forming a line connecting the 2 training points
            slope, intercept = get_target_function(trainingData[0], trainingData[1])
            prediction = slope*validationData[0] + intercept
            # inputMatrix = np.array([(1, dataPoint[0]) for dataPoint in trainingData])
            # not sure why one step learning doesn't work here. gives me all the same cross validation error right now
            # predictions, _, weights = LinearRegression(inputMatrix, [valueTarget[1] for valueTarget in trainingData], [validationData[1]])
            constantError = (validationData[1] - constantWeight)**2
            linearError = (validationData[1] - prediction)**2
            constantRoError.append(constantError)
            linearRoError.append(linearError)
        constantErrors.append((np.mean(constantRoError), ro, roIndex))
        linearErrors.append((np.mean(linearRoError), ro, roIndex))
    return constantErrors, linearErrors

def extractPts(inputs):
    """extract input values for (x, y) points into 'input' (1, x) and a target (y)"""
    x = []
    y = []
    for input in inputs:
        x.append((1, inputs[0]))
        y.append(inputs[2])
    return x, y


def getPredictions(weights, inputs):
    """get list of predictions"""
    predictions = []
    for input in inputs:
        predictions.append(np.sign(np.dot(weights, input)))
    return predictions

def getRandomPt(points):
    """helper function that gets a random (x, y) point from the argument"""
    randomIndex = randint(0, len(points) - 1)
    return points[randomIndex]

def PLA(inputPts):
    """implmentation of the simple perception learning algorithm. Picks a randomly misclassified point and updates the
    weights according to the equation found on pg 7 of the Learning From Data textbook (w(t+1) = w(t) + y(t)*x(t)"""
    def findMisclassfied(weights, xy, inputPts, targets):
        """helper functions that takes in weights and uses the current weight vector to make predictions to compare
        to the target values. Then outputs a list of points that are misclassified
        xy - the actual (x, y) points
        inputPts - the input points for the linear model ie (1, x) - 1 being the bias term
        targets - the y values for the (x, y) points"""
        predictions = getPredictions(weights, inputPts)
        assert len(predictions) == len(targets)
        misclassifiedPts = []
        for index, predict in enumerate(predictions):
            if predict != targets[index]:
                misclassifiedPts.append(xy[index])
        return misclassifiedPts

    weights = np.array([0, 0])
    inputX, inputY = extractPts(inputPts)
    wrongPts = findMisclassfied(weights, inputPts, inputX, inputY)
    while len(wrongPts) == 0:
        randomWrongPt = getRandomPt(wrongPts)
        weights += weights + randomWrongPt[1]*np.array([1, randomWrongPt[0]])
    return weights

def createDataSet(N=10):
    """helper that returns points (x, y, target) for some random target functiont that is created"""


constant, linear = problem7()
print(constant)
print(linear)


