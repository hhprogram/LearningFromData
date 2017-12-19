import numpy as np
from numpy.linalg import inv

def getWeights(inputs, targets, regLamba=None):
    """see slide15 of lecture 15 for linear regression one step learning. note: INPUTS should have the extra bias
    term coefficient of '1' added onto it. Therefore, if input space is R^2 then vectors in INPUTS should be in R^3
    also remember the input matrix INPUTS is structured s.t. each row is an specific input data point"""
    lambdaCoefficent = 0
    if regLamba is not None:
        lambdaCoefficent = regLamba
    xTx = np.matmul(inputs.T, inputs)
    identity = np.identity(xTx.shape[0])
    xTx = xTx + identity*lambdaCoefficent
    xTx_inverse = inv(xTx)
    x_cross = np.matmul(xTx_inverse, inputs.T)
    return np.matmul(x_cross, targets)


def nonLinearTransform(inputs):
    """assumes the bias coefficient of 1 has been added. and assumes the input space is in R^2
    transform given by formula in HW6"""
    zInput = []
    for input in inputs:
        x1 = input[1]
        x2 = input[2]
        z1 = x1
        z2 = x2
        z3 = x1**2
        z4 = x2**2
        z5 = x1*x2
        z6 = abs(x1 - x2)
        z7 = abs(x1 + x2)
        zInput.append([1, z1, z2, z3, z4, z5, z6, z7])
    return np.array(zInput)


def LinearRegression(zData, target, zOutput, regTerm=None, regLambda=0):
    """takes in training data points (assuming the bias coefficient of 1 is already added). Then it runs linear
    regression on it (one step learning). And if regTerm is not None then added a regularization term and then does
    linear regression to find weights (see slide 11, lecture 12 and see w(reg)
    assumes the input and output vectors have been transformed to Z space already if that is the desire"""
    if regTerm:
        w = getWeights(zData, target, regLambda)
    else:
        w = getWeights(zData, target, None)
    predictions = []
    e_in = []
    for count, train_pt in enumerate(zData):
        prediction = np.sign(np.dot(train_pt, w))
        if prediction != target[count]:
            e_in.append(1)
        else:
            e_in.append(0)
    for test_pt in zOutput:
        prediction = np.sign(np.dot(test_pt, w))
        predictions.append(prediction)
    return predictions, np.mean(e_in), w

def getTestError(predictions, targets):
    errors = []
    for count, predict in enumerate(predictions):
        if predict != targets[count]:
            errors.append(1)
        else:
            errors.append(0)
    return np.mean(errors)

def getPredictions(weights, zInput):
    """helper function that outputs an array of predictions given a vector of weights for some model. Assumes everything
    has been transformed to z space already"""
    predictions = []
    for pt in zInput:
        prediction = np.sign(np.dot(pt, weights))
        predictions.append(prediction)
    return predictions

def getRawData():
    """method gets the raw data from the training and testing data files. Then adds the bias coefficient to each
    input. Also, splits out the target and input data arrays and returns a 4 tuple"""
    trainData = np.loadtxt('/home/harrison/LearningFromDataCode/HW6in.dta')
    trainData = np.insert(trainData, 0, 1, axis=1)
    trainTarget = trainData[:, [3]]
    trainInput = trainData[:, [0, 1, 2]]
    testData = np.loadtxt('/home/harrison/LearningFromDataCode/HW6out.dta')
    testData = np.insert(testData, 0, 1, axis=1)
    testTarget = testData[:, [3]]
    testInput = testData[:, [0, 1, 2]]
    return trainInput, trainTarget, testInput, testTarget


# trainInput, trainTarget, testInput, testTarget = getRawData()
# trainInput = nonLinearTransform(trainInput)
# predictions, e_in, w = LinearRegression(trainInput, trainTarget, testInput)
# print(e_in, getTestError(predictions, testTarget)) # prints out the answer to number 2
#
# predictions, e_in, w = LinearRegression(trainInput, trainTarget, testInput, regTerm=True, regLambda=10**(-3))
# print(e_in, getTestError(predictions, testTarget)) #prints out the answer to number 3
#
# predictions, e_in, w = LinearRegression(trainInput, trainTarget, testInput, regTerm=True, regLambda=10**(3))
# print(e_in, getTestError(predictions, testTarget)) #prints out the answer to number 4
#
# predictions, e_in, w = LinearRegression(trainInput, trainTarget, testInput, regTerm=True, regLambda=10**(-1))
# print(e_in, getTestError(predictions, testTarget)) #prints out the answer to number 5