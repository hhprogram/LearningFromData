import numpy as np
from random import randrange, randint
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv

def get_target_function(pt1, pt2):
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    y_intercept = pt2[1] - slope*pt2[0]
    return slope, y_intercept

def evaluate_point(slope, y_intercept, pt):
    # anything above or touching line is +1 anything below is -1
    target = slope * pt[0] + y_intercept
    if pt[1] >= target:
        return 1
    elif pt[1] < target:
        return -1
    raise TypeError("??")

def get_weights(inputs, target_values):
    # uses the formula found in lecture 3 slide 15. target_values = vector of individual target values associated with
    # each row in INPUTS matrix
    input_array = np.array(inputs)
    xTx = np.matmul(input_array.T, input_array)
    xTx_inverse = inv(xTx)
    x_cross = np.matmul(xTx_inverse, input_array.T)
    return np.matmul(x_cross, target_values)

def PLA(num_weights, inputs, targets, initial=None):
    # perceptron learning algorithm. returns tuple of weight vector and number of iterations to learn
    # assumed the inputs and targets arrays are 'aligned' ie the 0th index in the targets array corresponds to the
    # 0th index item in the inputs array
    if initial is None:
        weights = [.5]*num_weights
        initial = np.array(weights)
    weights = initial
    g_slope = -weights[1] / weights[2]
    g_intercept = -weights[0] / weights[2]
    misclassified = [(pt[0], pt[1], pt[2], count) for count, pt in enumerate(inputs) if evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) != targets[count]]
    iterations = 0
    while len(misclassified) > 0:
        random_point = misclassified[randint(0, len(misclassified)-1)]
        weights += np.array((random_point[0], random_point[1], random_point[2])) * training_targets[random_point[3]]
        g_slope = -weights[1] / weights[2]
        g_intercept = -weights[0] / weights[2]
        misclassified = [(pt[0], pt[1], pt[2], count) for count, pt in enumerate(inputs) if
                         evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) != targets[count]]
        iterations += 1
    return weights, iterations

if __name__ == "__main__":
    e_in = []
    e_out = []
    gs = []
    # run this experiment 1000 times. questions 5 and 6 in hw 2
    for step in range(1000):
        # training points and their corresponding target values
        training_points = []
        training_targets = []
        N = 100
        # getting the slope and intercept of the target function
        slope, y_intercept = get_target_function((-1, random.uniform(-1, 1)), (1, random.uniform(-1, 1)))
        # generate the N training points and assigning them their target values
        for i in range(N):
            # puts the x(0) term in there for out weight offset term
            pt = (random.uniform(-1,1), random.uniform(-1,1))
            training_points.append((1, pt[0], pt[1]))
            training_targets.append(evaluate_point(slope, y_intercept, pt))
        # use linear regression to get the weight vector
        weight = get_weights(training_points, training_targets)
        # then convert the weights into slope intercept form
        g_slope = - weight[1] / weight[2]
        g_intercept = - weight[0] / weight[2]
        gs.append((g_slope, g_intercept))
        # evaluate the training data
        e_in.append(sum([evaluate_point(g_slope, g_intercept, (training_points[index][1], training_points[index][2])) != training_targets[index] for index in range(N)]) / N)
        out_points = []
        out_targets = []
        # generate 1000 'out' of sample data
        for i in range(1000):
            # puts the x(0) term in there for out weight offset term
            pt = (random.uniform(-1,1), random.uniform(-1,1))
            out_points.append((1, pt[0], pt[1]))
            out_targets.append(evaluate_point(slope, y_intercept, pt))
        # evaluate our estimation function on the 'out' of sample data
        e_out.append(sum([evaluate_point(g_slope, g_intercept, (out_points[index][1], out_points[index][2])) != out_targets[index] for index in range(1000)]) / 1000)

    print(np.mean(e_in))
    print(np.mean(e_out))

    # below is supposed to be question 7 in hw2
    N = 10
    iterations = []
    for i in range(1000):
        slope, y_intercept = get_target_function((-1, random.uniform(-1, 1)), (1, random.uniform(-1, 1)))
        training_targets = []
        training_points = []
        for _ in range(N):
            pt = (random.uniform(-1, 1), random.uniform(-1, 1))
            training_points.append((1, pt[0], pt[1]))
            training_targets.append(evaluate_point(slope, y_intercept, pt))
        weights = get_weights(training_points, training_targets)
        weights, num_iterations = PLA(3, training_points, training_targets, weights)
        print(i, num_iterations)
        iterations.append(num_iterations)
    print("Average number of iterations: ", np.mean(iterations))


    #### single example to visualize. helped me debug errors in the above larger experiment. Good notes below
    training_points = []
    training_targets = []
    # create the 2 points using (-1, some random value between (-1,1)) and (1, some random value between (-1,1))
    # did this vs. both x,y being random as then the line could actually out of the range and wouldn't really make sense
    # now did this to ensure the target function line was within the y range at both x ranges
    slope, y_intercept = get_target_function((-1, random.uniform(-1, 1)), (1, random.uniform(-1, 1)))
    for i in range(N):
        # puts the x(0) term in there for out weight offset term
        pt = (random.uniform(-1, 1), random.uniform(-1, 1))
        training_points.append((1, pt[0], pt[1]))
        training_targets.append(evaluate_point(slope, y_intercept, pt))
    weight = get_weights(training_points, training_targets)
    # the estimation function needs to negate each of the following 2 fractions because the weight vector is
    # wTx = y(n). which translates to w(0) + w(1)x + w(2)y = y(n) or the decision boundary is w(0) + w(1)x + w(2)y = 0
    # then solve for each slope and y-intercept of this equation
    g_slope = -weight[1] / weight[2]
    g_intercept = -weight[0] / weight[2]
    point1 = g_slope*-1 + g_intercept
    point2 = g_slope*1 + g_intercept
    plt.plot([-1,1], [point1, point2], 'b')
    # below are the iterables for target +1 and -1 values
    # need to take only positions 1,2 because the first position is a 'dummy' position used to just make matrix
    # operations possible for our weight vector calculations
    plus_one = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == 1]
    neg_one = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == -1]
    # the 2 lists used to then color each of the points green or red depending if the estimation function got it right
    wrong = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) != training_targets[count]]
    right = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) == training_targets[count]]
    # combine the logic to make 4 lists in order to then color code them to show estimation function right and wrong
    # and the point style is the target value indicator
    wrong_neg = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == -1 and evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) != training_targets[count]]
    wrong_pos = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == 1 and evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) != training_targets[count]]
    right_neg = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == -1 and evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) == training_targets[count]]
    right_pos = [(pt[1], pt[2]) for count, pt in enumerate(training_points) if training_targets[count] == 1 and evaluate_point(g_slope, g_intercept, (pt[1], pt[2])) == training_targets[count]]
    print(len(wrong_neg) + len(wrong_pos) + len(right_neg) + len(right_pos))
    plt.plot([pt[0] for pt in wrong_neg], [pt[1] for pt in wrong_neg], 'r^')
    plt.plot([pt[0] for pt in wrong_pos], [pt[1] for pt in wrong_pos], 'r+')
    plt.plot([pt[0] for pt in right_neg], [pt[1] for pt in right_neg], 'g^')
    plt.plot([pt[0] for pt in right_pos], [pt[1] for pt in right_pos], 'g+')
    point1 = slope*-1 + y_intercept
    point2 = slope*1 + y_intercept
    plt.plot([-1,1], [point1, point2],'k')
    plt.show()




