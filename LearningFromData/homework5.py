import numpy as np
# HW5 problems from Learning From Data

def problem5():
    # gradient descent on error function E(u, v) = (ue^v − 2ve^(−u))^2
    # first we calculated, by hand, that partial derivative of E with respect to u is
    # 2(e^v + 2ve^(−u))(ue^v − 2ve^(−u))
    # therefore, follow gradient descent algorithm of initializing weights then calculating gradient, then adding
    # that times step size, eta, to the previous weight and keep going.
    # initialize weight at 0 (which is ok with gradient descent)
    def computeError(u, v):
        return float((u*(np.e)**v - 2*v*(np.e)**(-u))**2)

    def gradientRespectWithU(u, v):
        # evaluates the gradient of E with respect to U at points (u,v)
        return 2*((np.e)**v + 2*v*(np.e)**(-u))*(u*(np.e)**v - 2*v*(np.e)**(-u))

    def gradientRespectWithV(u, v):
        # evaluates the gradient of E with respect to V at points (u,v)
        return 2 * (u*(np.e) ** v - 2 * v * (np.e) ** (-u)) * (u * (np.e) ** v - 2 * (np.e) ** (-u))

    def computeGradient(func):
        def gradient(u,v):
            return func(u, v)
        return gradient

    # made each one floats explicitly as when making the loop for problem 7 since it was ints it was rounding down
    # to 0 when it was a decimal
    u = 1.0
    v = 1.0
    w = np.array([u,v])
    eta = .1
    # compute initial gradient at point (u,v) = (1,1) with respect to U and V (do partial derivative for one so we know
    # how much to move for each variable)
    gradientU = computeGradient(gradientRespectWithU)
    gradientV = computeGradient(gradientRespectWithV)
    iterationCount = 0
    threshold = float(10**(-14))
    error = computeError(w[0], w[1])
    while error > threshold:
        # get partial derivative with respect to u and v evaluated at the new point
        gradient_vector = np.array([gradientU(w[0], w[1]), gradientV(w[0], w[1])])
        # update rule per slide 23 in lecture 9
        w = w - eta * gradient_vector
        iterationCount += 1
        error = computeError(w[0], w[1])
    #problem 5 & 6
    print(iterationCount, w[0], w[1])

    #problem 7
    count = 0
    w = np.array([u, v])
    while count < 15:
        w[0] = w[0] - eta * gradientU(w[0], w[1])
        w[1] = w[1] - eta * gradientV(w[0], w[1])
        error = computeError(w[0], w[1])
        count += 1
    print(error, w[0], w[1])


# problem5()

import random


def problem8():
    # goes over problem 8 and 9. Returns tuple first element is problem 8's answer and 2nd is problem 9
    def getLine(p1, p2):
        # calculates line given 2 points. Returns a function that takes in an 'x' value and gives the corresponding
        # y value. This is the border line of classifier. Therefore, anything above line is +1 and below is -1
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p2[1] - slope * p2[0]
        def func(x):
            return slope*x + intercept
        return func
    def cross_entropy(X, Y, w):
        errors = []
        for index, input in enumerate(X):
            error = np.log(1.0 + np.exp(-Y[index]*np.dot(w, input)))
            errors.append(error)
        return np.mean(errors)
    # forming the boundary line via 2 random points
    pt1 = (random.uniform(-1, 1), random.uniform(-1, 1))
    pt2 = (random.uniform(-1, 1), random.uniform(-1, 1))
    boundaryLine = getLine(pt1, pt2)
    trainingPts = 100
    # the list that will be shuffled to randomly walk through training points stochasticly
    permutation = [i for i in range(trainingPts)]
    trainingInputs = []
    trainingOutputs = []
    for i in range(trainingPts):
        pt = (random.uniform(-1, 1), random.uniform(-1, 1))
        if boundaryLine(pt[0]) < pt[1]:
            trainingOutputs.append(1)
        else:
            trainingOutputs.append(-1)
        trainingInputs.append(pt)
    w = np.array([0.0, 0.0, 0.0])
    eta = .01
    epochCount = 0
    # shuffling the indices list mimics stochastic choice
    random.shuffle(permutation)
    iterationCount = 0
    weight_change = 100
    while weight_change > .01:
        w_prev = w
        if iterationCount == 0:
            w_prev_epoch = w_prev
        index = permutation[iterationCount]
        # go through each N training points in each epoch see slide 23 of lecture 9 for gradient eq.
        # do it one by one therefore each loop in while loop looks at one pt. It's 'stochastic' because we have
        # shuffled the points
        # add the extra '1' as the x_0 constant term
        # remember the resulting gradient of Ein is a vector with same dimension as 'w' vector since I am
        # adding it, scaled by eta, to 'w' vector
        outputValue = trainingOutputs[index]
        inputVector = np.array([1, trainingInputs[index][0], trainingInputs[index][1]])
        numerator = -outputValue * inputVector
        exp = outputValue * (np.dot(w, inputVector))
        denominator = 1 + np.exp(exp)
        summationTerm = numerator / denominator
        gradientEin = summationTerm
        w = w_prev - eta * gradientEin
        if iterationCount < len(permutation) - 1:
            iterationCount += 1
        else:
            iterationCount = 0
            epochCount += 1
            weight_change = np.linalg.norm(w_prev_epoch - w)
    testingPts = 1000
    # for i in range(testingPts):
    #     pt = (random.uniform(-1, 1), random.uniform(-1, 1))
    #     # making use of ternary operators
    #     trueOutput = 1 if boundaryLine(pt[0]) < pt[1] else -1
    #     # count it as an error is the signs are different ie trueOutput != prediction therefore the product of the
    #     # 2 will be negative. Else it is categorized correctly
    #     # eout = 1 if trueOutput*np.dot(w,np.array([1, pt[0], pt[1]])) < 0 else 0
    #     eOuts.append(eout)
    testingPtsList = []
    trueOutputs = []
    for i in range(testingPts):
        pt = np.array([1, random.uniform(-1, 1), random.uniform(-1, 1)])
        testingPtsList.append(pt)
        # truevalue = 1 if boundaryLine(pt[0]) < pt[1] else -1
        trueOutputs.append(1 if boundaryLine(pt[1]) < pt[2] else -1)

    return cross_entropy(testingPtsList, trueOutputs, w), epochCount

eOuts = []
iterations = []
for i in range(100):
    eout, iterationCount = problem8()
    eOuts.append(eout)
    iterations.append(iterationCount)
print(np.mean(eOuts), np.mean(iterations))