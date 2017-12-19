#HW4 problem 2 from Learning From Data
import numpy as np
from scipy import optimize
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv

def graph_bounds(delta=.05, d_vc=50):
    # graphs the bounds on epsilon as seen in problem 2 HW4
    for N in np.arange(1, 10000, 1000):
        originalTerm = (8 / N) * np.log((4*((2*N)**d_vc)) / delta)
        originalBound = np.sqrt(originalTerm)
        plt.plot(N, originalBound, 'k')
        rademacherBound = np.sqrt((2*np.log((2*N*((2*N)**d_vc)))) / N) + np.sqrt((2 / N) * np.log(1 / delta)) + 1 / N

        plt.plot(N, rademacherBound, 'b')
        # def parrondoRHS(epsilon):
        #     return epsilon - np.sqrt((1 / N) * (2 * epsilon + np.log((6 * ((2*N)**(d_vc))) / delta )))
        # parrondoBound = optimize.brentq(parrondoRHS, -10, 1)
        # plt.plot(N, parrondoBound, 'r')
        # def devroye(epsilon):
        #     return epsilon - np.sqrt( (1/ (2*N)) * ((4*epsilon)*(1 + epsilon)) + np.log((4 * ((N**2)**d_vc)) / delta))
        # devroyeBound = optimize.brentq(devroye, -10, 1)
        # plt.plot(N, devroyeBound, 'y')

# graph_bounds()

def findGBar(N=100000):
    # question 4 homework 4 Learning from Data
    # target function is sin(pi*x). Our training data is 2 x values randomly selected between -1 and 1.
    # our hypothesis set contains all solutions of the form h(x) = ax. Therefore, returns the average of a (which is
    # 'weight' over N iterations
    weights = []
    for i in range(N):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x_vector = np.array([[x1], [x2]])
        y1 = np.sin(np.pi*x1)
        y2 = np.sin(np.pi*x2)
        y_vector = np.array([[y1], [y2]])
        #then use lecture 3 slide 15 to find easy way to get the least mean squared error values
        xTx_inv = 1 / np.matmul(x_vector.T, x_vector)
        x_cross = xTx_inv * x_vector.T
        weight = np.matmul(x_cross, y_vector)
        weights.append(weight)
    return np.mean(weights)


def findBias(gBar, N=100000):
    # question 5 hw 4 from Learning From Data. Finds the bias via the definiton from lec 8 slide 9. Expected value
    # (over the input space) of the squared difference between gbar(x) and f(x). f(x) -> target function. gbar(x)
    # is my best approx (given a certain hypothesis set) of the target
    squaredErrors = []
    for i in range(N):
        x = random.uniform(-1, 1)
        squaredError = (gBar*x - np.sin(np.pi*x))**2
        squaredErrors.append(squaredError)
    return np.mean(squaredErrors)

def findVariance(N=100000):
    # question 6 HW4 from Learning From Data. Note: see lecture 8 slide 9 - the for loop I actually switch the order of
    # the expected values (and I can do that because the inner function is always positive. I guess you can switch
    # integral orders if the integrand is always positive). I did that as it was a little simpler for me to understand
    expectedDataSetErrors = []
    gbar = findGBar(N)
    for ii in range(int(N / 100)):
        specificG = findGBar(N=1)
        squaredErrors = []
        for i in range(N):
            x = random.uniform(-1, 1)
            squaredError = (specificG*x - gbar*x)**2
            squaredErrors.append(squaredError)
        expectedErrorDataset = np.mean(squaredErrors)
        expectedDataSetErrors.append(expectedErrorDataset)
    return np.mean(expectedDataSetErrors)


hypotheses = {'constant': [1, 0], 'linear_noIntercept': [0, 1], 'linear_Intercept': [1, 1]}

def findOutOfSampleErrors(N=100000):
    # for problem 7 Hw 4 from Learning From Data

    def _findgBarOneWeight(N=100000, intercept=True, transform=False):
        # needed this one as since there were some hypothesis sets of form h(x) = b or h(x) = ax, then we are dealing
        # with input vectors that will actually just be one number either 1 if just = b or x is ax. Therefore,
        # the inverse will not be an inverse of a matrix but just of a number
        weights = []
        transform_factor = 1
        if transform:
            transform_factor = 2
        for _ in range(N):
            x1 = random.uniform(-1, 1)**transform_factor
            x2 = random.uniform(-1, 1)**transform_factor
            y1 = np.sin(np.pi * x1)
            y2 = np.sin(np.pi * x2)
            if intercept:
                x_vector = np.array([1, 1])
            else:
                x_vector = np.array([x1, x2])
            y_vector = np.array([y1, y2])
            xTx_inv = 1 / np.matmul(x_vector.T, x_vector)
            x_cross = xTx_inv * x_vector.T
            weight = np.matmul(x_cross, y_vector)
            weights.append(weight)
        return np.mean(weights)


    def _findGBar(hyp_form, N=100000, transform=False):
        # transform boolean indicates whether or not I want to 'transform' my input by squaring it or leave it as is
        # finds gBar by going N times and then each iteration uses the pseudo-inverse 1 step learning (seen in lecture 3
        # and then finds the optimal weights and then after all the N iterations I take the average of all weights
        # and those weights are associated with my gBar function
        weights = []
        transform_factor = 1
        if transform:
            transform_factor = 2
        for _ in range(N):
            original_x1 = random.uniform(-1, 1)
            original_x2 = random.uniform(-1, 1)
            x1_vector = [1, original_x1**transform_factor]
            x2_vector = [1, original_x2**transform_factor]
            y1 = np.sin(np.pi * original_x1)
            y2 = np.sin(np.pi * original_x2)
            y_vector = np.array(y1, y2)
            x_matrix = np.array([x1_vector, x2_vector])
            xTx = np.matmul(x_matrix.T, x_matrix)
            x_cross = np.matmul(inv(xTx), x_matrix.T)
            weight = np.multiply(x_cross, y_vector)
            weights.append(weight)
        # do this in order to take the mean within the columns
        return np.mean(weights, axis=0)
    # gBars is a dict with key of the string of the specific hypothesis set and value is a 2 element list
    # [gBar weight vector, out of sample error]
    gBars = {}
    # going through each hypothesis type and calculating their gBars
    for hypothesis in hypotheses:
        if hypothesis == 'constant':
            gBars[hypothesis] = [[_findgBarOneWeight(intercept=True), 0], None]
        if hypothesis == 'linear_noIntercept':
            gBars[hypothesis] = [[0, _findgBarOneWeight(intercept=False)], None]
            gBars['transform'+hypothesis] = [[0, _findgBarOneWeight(intercept=False, transform=True)], None]
        else:
            gBars[hypothesis] = [_findGBar(hypothesis), None]
            gBars["transformed" + hypothesis] = [_findGBar(hypothesis, transform=True), None]
    # then for each gBar function I generate more points (my out of sample points) and then test each gBar on these
    # points and calculate the error to see which gBar achieves the lowest out of sample error
    for gBar in gBars:
        errors = []
        biases = []
        variances = []
        for _ in range(N):
            x = random.uniform(-1, 1)
            y = np.sin(np.pi * x)
            weight_vector = gBars[gBar][0]
            est_y = np.dot(weight_vector, [1, x])
            error = (y - est_y)**2
            errors.append(error)
            bias = (est_y - y)**2
        gBars[gBar][1] = np.mean(errors)
        print(gBar, gBars[gBar][1])

#uncomment below to show the answers that the corresponding methods solve for
# gBar = findGBar()
# print(gBar)
# print(findBias(gBar))
# print(findVariance())
# findOutOfSampleErrors()

def equation(N):
    # problem number 1 for HW4 Learnign from Data Use this by plugging in some N and keep plugging in higher values
    # of N until this returns a satifactory 1 - delta, such that 1-delta is the % confidence (ie delta is the % times
    # that out of sample error will be above some threshold (which in this case we have set to .05)
    # see Lecture 7 slide 23 for the equation. And then we are solving for delta and plugging in (2N)^(VC dimension) in
    # for the growth function
    return 4 * (2*N)**(10) * np.e**((-1 / 8)* ((.05)**2) * N)

print(equation(452957))