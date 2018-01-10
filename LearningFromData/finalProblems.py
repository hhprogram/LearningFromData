import numpy as np
import random
from hw8Kernels import getRawData, oneVersusMany, oneVersusOne, SVMclassifier
from typing import List, Tuple, Union
import matplotlib.pyplot as plt

def polyTransform(inputData: Union[List, Tuple, np.ndarray], q=2):
    """helper function that transforms the INPUTDATA to a z space using a polynomial transform of order Q
    see an example in HW5 page3 right above problem3 (polynomial transform of order 4)
    """
    if q is None:
        return inputData
    zInput = []
    for inputPt in inputData:
        newPt = []
        for i in range(q+1):
            for j in range(q+1):
                if i+j <= q:
                    newPt.append((inputPt[0]**i) * inputPt[1]**j)
        zInput.append(newPt)
    return zInput

def addBiasTerm(inputData: Union[List, Tuple, np.ndarray], biasTermValue=1):
    """helper function that adds the intercept/bias term in every input vector"""
    # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.insert.html
    inputWithBias = np.insert(inputData, 0, 1, axis=1)
    return inputWithBias

class LinearClassifier:
    def __init__(self, zInput: Union[List, Tuple, np.ndarray], targets: Union[List, Tuple], classes: Union[List, Tuple],
                 one,
                 many: Union[bool, str, int, float]=True, lambdaParameter: Union[float, int]=1):
        """to be used for problems 7,8,9,10 of the final. Assumes the inputs have be transformed to whatever desired z
        space. see slide 11 in lecture 12 for the solution. Linear regression pseudo inverse solution but just with
        regularization (the lambda*I is the part that adds the regularization)
        MANY is either TRUE if it is one versus many and then the other attribute just takes on a list of classes
        derived from CLASSES all except for the ONE attribute. If it isn't true then it takes in the class that other
        should be for one versus one"""
        if type(zInput) != np.ndarray:
            zInput = np.array(zInput)
        zTz = np.matmul(zInput.T, zInput)
        dimension = zTz.shape[0]
        identityMatrix = np.identity(dimension)*lambdaParameter
        zTzInverse = np.linalg.inv(zTz + identityMatrix)
        wCross = np.matmul(zTzInverse, zInput.T)
        self.wReg = np.matmul(wCross, targets)
        self.one = one
        if many is True:
            classes.remove(self.one)
            self.other = classes
        else:
            self.other = many


    def getPredictions(self, zInput: Union[List, Tuple, np.ndarray]):
        predictions = []
        for inputPt in zInput:
            predictions.append(np.sign(np.dot(self.wReg, inputPt)))
        return predictions

    def checkIfTransformed(self, targets: Union[List, Tuple]):
        """helper function that checks if target data has be transformed to a binary format (ie one versus or one
        versus many"""
        # values = set(targets) #note cannot do this as ndarray is not hashable and thus cannot be easily converted
        # into a set with this construction
        transformedTargets = []
        for index, target in enumerate(targets):
            if not (-1 == target or 1 == target):
                for target in targets:
                    if target == self.one:
                        transformedTargets.append(1)
                    else:
                        transformedTargets.append(-1)
            if index == len(targets) - 1 and len(transformedTargets) == 0:
                return targets
        return transformedTargets

    def getError(self, zInput: Union[List, Tuple, np.ndarray], targets: Union[List, Tuple]):
        predictions = self.getPredictions(zInput)
        errors = []
        targets = self.checkIfTransformed(targets)
        for index, predict in enumerate(predictions):
            if predict == targets[index]:
                errors.append(0)
            else:
                errors.append(1)
        return np.mean(errors)

def makeAllOneVersusManys(inputs: Union[List, Tuple, np.ndarray],
                          orginalTargets: Union[List, Tuple],
                          orderPolyTransform=None,
                          lambdas: Union[List, Tuple, int, float]=1):
    """helper function that returns a list of classifiers. for 0 versus all,...., 9 versus all in order"""
    classifiers = []
    classes = [i for i in range(10)]
    for i in range(10):
        targets = oneVersusMany(orginalTargets, i)
        zInputs = polyTransform(inputs, orderPolyTransform)
        classifier = LinearClassifier(zInputs, targets, classes, i, lambdaParameter=lambdas)
        classifiers.append(classifier)
    return classifiers

def euclideanDistance(pt1, pt2):
    import math
    x1 = pt1[0]
    x2 = pt1[1]
    d1 = pt2[0]
    d2 = pt2[1]
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# object that acts as a centroid. has attributes members which is a list of points belonging to it and the location of
# it
class Centroid:
    def __init__(self, location: Tuple, members: List):
        self.location = location
        self.members = members

    def getLocation(self):
        return self.location

    def getMembers(self):
        return self.members

    def addMember(self, newMember):
        self.members.append(newMember)

    def moveLocation(self, newLocation):
        self.location = newLocation

    def removeMember(self, member):
        self.members.remove(member)

    def emptyCluster(self):
        if len(self.members) == 0:
            return True
        return False

def lloyd(inputs, k):
    """runs lloyd's algorithm. K is the number of centers we will be using. See slide 11 of
    lecture 16. Returns a list of the centers that lloyd's algorithm has converged to. Last element of the tuple
    is FALSE if there is a cluster with no members. See while loop for meat of the algorithm. First keep centroids
    constant and just optimize grouping of points to centroids closest to them. And then, hold subgroups constant
    and then update the centroids by finding the average of the latest groupings"""
    import copy
    indices = list(range(len(inputs) - 1))
    random.shuffle(indices)
    centers = []
    subGroups = {}
    # initial creation of centers (just picking random K number of inputs to start as the centers)
    for index in range(k):
        centers.append(inputs[index])
        # each subgroup will be a centroid object that has info on its location and the members that belong to it
        subGroups[index] = Centroid(centers[index], [])
    # below we start assign all the input points to the last indexed centroid
    startCentroid = subGroups[index]
    [startCentroid.addMember(point) for point in inputs]
    # not sure if this is best, but added this to make it an easy mapping of input point back to centroid
    newInputs = [[point, index] for point in inputs]
    oldCenters = []
    # iteratively loop through centers.
    # boolean to see if the centroids have moved. Once they haven't moved then we break out of while loop
    changedFromLastTime = True
    count = 0
    while changedFromLastTime:
        # loop through each input point to reassign input points to the closest centers (ie re organizing subgroups)
        for inputIndex, inputInfo in enumerate(newInputs):
            inputPt, groupID = inputInfo
            minDistance = float('inf')
            subGroups[groupID].removeMember(inputPt)
            for index, center in enumerate(centers):
                distance = euclideanDistance(inputPt, center)
                if distance < minDistance:
                    minDistance = distance
                    newCenterIndex = index
            newInputs[inputIndex][1] = newCenterIndex
            subGroups[newCenterIndex].addMember(inputPt)
        # loop through each new subgroup and create new centroids. First copy previous centroids to a different object
        # list so we can check if the centroids moved between iterations
        oldCenters = copy.copy(centers)
        for index, _ in enumerate(centers):
            if subGroups[index].emptyCluster():
                newCenter = subGroups[index].getLocation()
            else:
                newCenter = np.mean(subGroups[index].getMembers(), axis=0)
            subGroups[index].moveLocation(newCenter)
            centers[index] = newCenter
        changedFromLastTime = False
        for index, _ in enumerate(centers):
            # because they are numpy arrays cannot just do centers == oldCenters (numpy arrays
            # do not allow for simple boolean checks on arrays. Need to check each element
            if oldCenters[index][0] != centers[index][0] or oldCenters[index][1] != centers[index][1]:
                changedFromLastTime = True
                break
        count += 1

    nonEmptyCluster = True
    for groupID in subGroups:
        if subGroups[groupID].emptyCluster():
            nonEmptyCluster = False
            break
    return subGroups, centers, nonEmptyCluster

# RBF classifier. uses like of CENTROIDS found by Lloyd's algorithm. And then we use slide 14 from lecture 16 to find
# the pseudo inverse vector to find the optimal weights to solve the equation (note the approx y(n) because so many
# equations and so few unknowns, must likely won't be able to find an exact solution.)
class RBFclassifer:
    def __init__(self, centroids: List, inputs, labels, gamma=1.5):
        self.centroids = centroids
        self.trainInputs = inputs
        self.trainTargets = labels
        self.gamma = gamma
        self.weights = self._findWeights()

    def _findWeights(self):
        """see slide 14 of lecture 16. Takes the centroids that we used Lloyd's algorithm to find, to then find the
        optimal weights to be used as a classifier. This determines the weights that should be associated with each
        centroid and determines if a point is a certain distance from some centroid how much influence should that
        centroid give to pushing that input point to a +1 / -1 label"""
        psi = []
        # loop through all the input point
        for point in self.trainInputs:
            # row is a list that represents a row in the psi matrix. Ie each row is of length K
            row = []
            # then to create each row for each input point loop through each centroid and execute the formula seen on
            # slide 14
            for centroid in self.centroids:
                l2NormSq = euclideanDistance(centroid, point) ** 2
                row.append(np.exp(-self.gamma * l2NormSq))
            psi.append(row)
        psi = np.array(psi)
        psiTpsi = np.matmul(psi.T, psi)
        psiTpsiInv = np.linalg.inv(psiTpsi)
        psiCross = np.matmul(psiTpsiInv, psi.T)
        return np.matmul(psiCross, self.trainTargets)

    def predict(self, point):
        # the prediction (hypothesis) is the dot product of the k weights vector and the vector whose elements are
        # exp(gamma * l2norm(centroid-k, point)) ...note in the prediction we need to transform the data in the form
        # that we saw in the psi matrix. The zInput for each x is a row within the psi matrix. ie look at slide 17 for
        # the equation h(x). that is the hypothesis equation for some input point 'x' that we want to classify.
        # therefore the input vector is turned into a k-length vector with each element equaling
        #  exp(-gamma * l2normsquare(x - centroid(k))
        zInput = [np.exp(-self.gamma * euclideanDistance(centroid, point)**2) for centroid in self.centroids]
        return np.sign(np.dot(self.weights, zInput))

    def getError(self, inputs, targets):
        """returns the error percentage on INPUTS and their corresponding targets"""
        errors = []
        for index, point in enumerate(inputs):
            if self.predict(point) == targets[index]:
                errors.append(0)
            else:
                errors.append(1)
        return np.mean(errors)

def targetFunction(inputPt):
    return np.sign(inputPt[1] - inputPt[0] + .25 * np.sin(np.pi * inputPt[0]))

def createDataSet(pts=100, target=targetFunction):
    """helper function that creates N number of data points according to the target function TARGET. Returns
    the input points and their +1/-1 value as two seperate lists"""
    X = []
    Y = []
    for _ in range(pts):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        X.append([x1, x2])
        label = target((x1, x2))
        Y.append(label)
    return X, Y


"""getting raw data and running final problem methods below"""

testInput, testTargetRaw, trainInput, trainTargetRaw = getRawData()

def problem7():
    inputWithBias = addBiasTerm(trainInput)
    classifiers = makeAllOneVersusManys(inputWithBias,
                                        trainTargetRaw)
    classfierInSampleErrors = []
    for classifier in classifiers:
        classfierInSampleErrors.append(classifier.getError(inputWithBias, trainTargetRaw))
    return classfierInSampleErrors

def problem8():
    classifiers = makeAllOneVersusManys(trainInput,
                                        trainTargetRaw,
                                        orderPolyTransform=2)
    classfierOutSampleErrors = []
    zTestInput = polyTransform(testInput, q=2)
    for classifier in classifiers:
        classfierOutSampleErrors.append(classifier.getError(zTestInput, testTargetRaw))
    return classfierOutSampleErrors

def problem9():
    inputWithBias = addBiasTerm(trainInput)
    testInputWithBias = addBiasTerm(testInput)
    normalClassifiers = makeAllOneVersusManys(inputWithBias,
                                        trainTargetRaw)
    transformClassifiers = makeAllOneVersusManys(trainInput,
                                        trainTargetRaw,
                                        orderPolyTransform=2)
    classfierOutSampleErrors = []
    zTestInput = polyTransform(testInput, q=2)
    zTrainInput = polyTransform(trainInput, q=2)
    for index in range(len(normalClassifiers)):
        classfierOutSampleErrors.append((normalClassifiers[index].getError(inputWithBias, trainTargetRaw),
                                         normalClassifiers[index].getError(testInputWithBias, testTargetRaw),
                                         transformClassifiers[index].getError(zTrainInput, trainTargetRaw),
                                         transformClassifiers[index].getError(zTestInput, testTargetRaw)))
    return classfierOutSampleErrors

def problem10():
    newInput, newTargets = oneVersusOne(trainInput, trainTargetRaw, 1, 5)
    newInput = polyTransform(newInput, q=2)
    zTestInput, zTestTarget = oneVersusOne(testInput, testTargetRaw, 1, 5)
    zTestInput = polyTransform(zTestInput, q=2)
    classifiers = []
    # note: don't need to actually even use the optional MANY argument because the inputs and targets have been filtered
    # to accomodate the one versus one paradigm
    classifiers.append(LinearClassifier(newInput, newTargets, [i for i in range(10)], 1))
    classifiers.append(LinearClassifier(newInput, newTargets, [i for i in range(10)], 1, lambdaParameter=.01))
    # list is [(e(in, lambda=1), e(out, lambda=1), (e(in, lambda=.01), e(out, lambda=.01)]
    answerChoices = []
    answerChoices.append(classifiers[0].getError(newInput, newTargets))
    answerChoices.append(classifiers[0].getError(zTestInput, zTestTarget))
    answerChoices.append(classifiers[1].getError(newInput, newTargets))
    answerChoices.append(classifiers[1].getError(zTestInput, zTestTarget))
    return answerChoices

def problem11():
    dataSet = [(1,0,-1), (0, 1, -1), (0,-1,-1), (-1,0,1), (0, 2, 1), (0,-2, 1), (-2,0,1)]
    zDataSet = np.array([(x[1]**2 - 2 * x[0] - 1, x[0]**2 - 2 * x[1] + 1, x[2]) for x in dataSet])
    positivesX = [x[0] for x in zDataSet if x[2] == 1]
    positivesY = [x[1] for x in zDataSet if x[2] == 1]
    negativesX = [x[0] for x in zDataSet if x[2] == -1]
    negativesY = [x[1] for x in zDataSet if x[2] == -1]
    print(len(positivesX) + len(negativesX))
    plt.scatter(positivesX, positivesY, color='g')
    plt.scatter(negativesX, negativesY, color='r')
    # given the scatter plot you can make a line and then solve for each value (see my paper for full solution)
    plt.show()

def problem12():
    inputs = [(1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), (-2, 0)]
    targets = [-1, -1, -1, 1, 1, 1, 1]
    SVM = SVMclassifier(inputs, targets, C=float('inf'), Q=2)
    # should be around 5 support vectors
    return SVM.getNumSupportVectors()

def problem13():
    X, Y = createDataSet()
    SVM = SVMclassifier(X, Y, C=float('inf'), kernel='rbf', gamma=1.5)
    eIn = SVM.getError(X, Y)
    # for problem 13 this should stay to be just zero or very close to zero
    nonSeperable = 0
    if eIn > 0:
        nonSeperable += 1
    return nonSeperable

def problem14And15():
    # for problem 14 set k=9, for problem 15 set k = 12
    k = 12
    gamma = 1.5
    kernelFormWins = []
    count = 0
    N=5000
    for _ in range(N):
        count += 1
        if count % 1000 == 0:
            print(count / N)
        X,Y = createDataSet()
        SVM = SVMclassifier(X, Y, C=float('inf'), kernel='rbf', gamma=gamma)
        if SVM.getError(X, Y) > 0:
            continue
        centroids, kCenters, nonEmptyCluster = lloyd(X, k)
        if nonEmptyCluster == False:
            continue
        RBF = RBFclassifer(kCenters, X, Y, gamma=gamma)
        testX, testY = createDataSet()
        svmTestError = SVM.getError(testX, testY)
        rbfError = RBF.getError(testX, testY)
        if svmTestError < rbfError:
            kernelFormWins.append(1)
        else:
            kernelFormWins.append(0)

    return np.mean(kernelFormWins)

def problem16and17():
    N=5000
    # for problem 16 set both gammas to 1.5. For problem 17 set 'gamma' to 1.5 and 'gamma2' to 2.
    gamma = 1.5
    gamma2 = 2
    # for problem 16 set k1 to 9 and k2 to 12
    k1 = 9
    k2 = 9
    count = 0
    answerchoicea = []
    answerchoiceb = []
    answerchoicec = []
    answerchoiced = []
    answerchoicee = []
    for _ in range(N):
        count += 1
        if count % 1000 == 0:
            print(count / N)
        X, Y = createDataSet()
        centroids12, kCenters12, nonEmptyCluster12 = lloyd(X, k2)
        centroids9, kCenters9, nonEmptyCluster9 = lloyd(X, k1)
        if nonEmptyCluster9 == False or nonEmptyCluster12 == False:
            continue
        rbfk9 = RBFclassifer(kCenters9, X, Y, gamma=gamma)
        rbfk12 = RBFclassifer(kCenters12, X, Y, gamma=gamma2)
        eIn9 = rbfk9.getError(X,Y)
        eIn12 = rbfk12.getError(X, Y)
        testX, testY = createDataSet()
        eOut9 = rbfk9.getError(testX,testY)
        eOut12 = rbfk12.getError(testX, testY)
        if eIn12 < eIn9 and eOut12 > eOut9:
            answerchoicea.append(1)
        if eIn12 > eIn9 and eOut12 < eOut9:
            answerchoiceb.append(1)
        if eIn12 > eIn9 and eOut12 > eOut9:
            answerchoicec.append(1)
        if eIn12 < eIn9 and eOut12 < eOut9:
            answerchoiced.append(1)
        if eOut12 == eIn9 and eOut12 == eOut9:
            answerchoicee.append(1)
    # prints out the number of occurences of each answerchoice below.
    # for problem 16 Should be that answerchoiced has the most
    # occurences, then answerchoicea, then the rest are quite similar with answerchoicee being the least frequent
    # for problem 17 should be that answerchoicec has the most occurences
    # then answerchoicee is next and then answerchoicea and b are about the same. then the remaining one has the least
    return(np.sum(answerchoicea),
          np.sum(answerchoiceb),
          np.sum(answerchoicec),
          np.sum(answerchoiced),
          np.sum(answerchoicee))

def problem18():
    gamma = 1.5
    k = 9
    N = 5000
    count = 0
    eInZeros = []
    for _ in range(N):
        count += 1
        if count % 1000 == 0:
            print(count / N)
        X, Y = createDataSet()
        centroids, kCenters, nonEmptyCluster = lloyd(X, k)
        if nonEmptyCluster == False:
            continue
        rbf = RBFclassifer(kCenters, X, Y, gamma=gamma)
        eIn = rbf.getError(X, Y)
        if eIn == 0:
            eInZeros.append(1)
    # should be answer choice a. one run said 2.4% of the time the rbf classifier is able to get zero in sample errors
    return np.sum(eInZeros) / N


if __name__ == '__main__':
    # prints out the in sample errors for 0 versus all, ..., 9 versus all
    # print("In sample errors: without transform")
    # print(problem7())
    # # prints out the in sample errors for 0 versus all, ..., 9 versus all
    # print("Out sample errors: with transform")
    # print(problem8())
    # prints out tuple of (E(in) 0 versus all (no transformed input), E(out) 0 versus all (transformed input),
    #                      E(in) 0 versus all (transformed input), E(out) 0 versus all (transformed input)),...,)
    # print(problem9())
    # below is the print out of problem 9. Because running it takes awhile. can see that E(out) not really impacted
    # significantly and using these values will reach the conclusion of (e) for problem 9
    # [(0.10931285146070498, 0.11509715994020926, 0.10231792621039638, 0.10662680617837568),
    #  (0.015224249074201069, 0.022421524663677129, 0.012343985735838706, 0.021923268560039861),
    #  (0.10026059525442327, 0.098654708520179366, 0.10026059525442327, 0.098654708520179366),
    #  (0.090248251268687421, 0.082710513203786751, 0.090248251268687421, 0.082710513203786751),
    #  (0.089425318886298177, 0.099651220727453915, 0.089425318886298177, 0.099651220727453915),
    #  (0.076258400768070222, 0.079720976581963129, 0.076258400768070222, 0.079222720478325862),
    #  (0.091071183651076665, 0.084703537618335822, 0.091071183651076665, 0.084703537618335822),
    #  (0.088465231106844053, 0.073243647234678619, 0.088465231106844053, 0.073243647234678619),
    #  (0.074338225209161987, 0.082710513203786751, 0.074338225209161987, 0.082710513203786751),
    #  (0.088328075709779186, 0.088191330343796712, 0.088328075709779186, 0.088191330343796712)]
    # see the answer choices. By the given values you can tell that overfitting occurs when going from lambda = 1
    # to lambda = .01 as the E(in) goes down when we go from lambda =1 model to the lambda =.01 model but then
    # the e(out goes up which implies overfitting.
    # print(problem10())
    # problem11()
    # print(problem12())
    # print(problem13())
    # problem 14 - should be the rbf kernel version wins most of the time. one run said about 78%.
    # also does problem 15 if you change the K value to 12 (for problem 15 got one run that was about 65%)
    # print(problem14And15())
    # print(problem16and17())
    print(problem18())