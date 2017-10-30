import numpy as np
from random import randrange, randint
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv

def genPoints(x1Low, x1High, x2Low, x2High, numPoints):
    # generates random points for 2d inputs
    points = []
    for _ in range(numPoints):
        points.append((1, random.uniform(x1Low, x1High), random.uniform(x2Low, x2High)))
    return points

def applyNoise(percentage, outputs):
    # takes percentage of the outputs and flips their signs. Outputs are values from a given target function
    numToFlip = int(percentage * len(outputs))
    startIndex = randint(0, len(outputs) - numToFlip)
    index = startIndex
    for _ in range(numToFlip):
        outputs[index] = - outputs[index]
        index += 1
    return outputs

def targetFunc(x1, x2, target=True, w0=None, w1=None, w2=None):
    # x0 defaults to the .6 value which is the target function value constant
    if target:
        return np.sign(x1**2 + x2**2 - .6)
    else:
        # print("hello")
        return np.sign(w0 + w1*x1 + w2*x2)

def nonLinearTransform(pt):
    x0 = pt[0]
    x1 = pt[1]
    x2 = pt[2]
    return (x0, x1, x2, x1*x2, x1**2, x2**2)

def nonLinearFunc(pt, w0, w1, w2, w3, w4, w5):
    x0 = pt[0]
    x1 = pt[1]
    x2 = pt[2]
    return np.sign(w0*x0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*(x1**2) + w5*(x2**2))

def getNonLinearFunc(w0, w1, w2, w3, w4, w5):
    def func(pt):
        x0 = pt[0]
        x1 = pt[1]
        x2 = pt[2]
        return np.sign(w0*x0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*(x1**2) + w5*(x2**2))
    return func

def weightVector(inputs, targets):
    x = np.array(inputs)
    xTx = np.matmul(x.T, x)
    xtx_inverse = inv(xTx)
    x_cross = np.matmul(xtx_inverse, x.T)
    return np.matmul(x_cross, np.array(targets))

N = 1000
e_in = []
# linear regression for the circular data. the e_in mean is about 50%. for question 8 in hw 2
for _ in range(N):
    points = genPoints(-1,1,-1,1, N)
    outputs = [targetFunc(pt[1], pt[2]) for pt in points]
    outputs_noise = applyNoise(.1, outputs)
    weights = weightVector(points, outputs_noise)
    errors = []
    for i in range(N):
        estimate = targetFunc(points[i][1], points[i][2], target=False, w0=weights[0], w1=weights[1], w2=weights[2])
        errors.append(not estimate == outputs_noise[i])
    errors = sum(errors)
    # for some reason below list comprehension not working. the func call goes to Target=True even though I have inputed it as False
    # errors = sum([not(func(points[count][1], points[count][2], target=False, w0=weights[0], w1=weights[1], w2=weights[2]) == outputs_noise[count] for count in range(len(points)))])
    e_in.append(errors / N)

print(np.mean(e_in))


# below are the set functions that we are comparing against in HW2 question 9:
functions = [getNonLinearFunc(-1, -.05, .08, .13, 1.5, 1.5),
             getNonLinearFunc(-1, -.05, .08, .13, 1.5, 15),
             getNonLinearFunc(-1, -.05, .08, .13, 15, 1.5),
             getNonLinearFunc(-1, -1.5, .08, .13, .05, .05),
             getNonLinearFunc(-1, -.05, .08, 1.5, .15, .15)]

e_in = []
# transform the data into nonlinear point and then to a linear regression on it
transformed_points = [nonLinearTransform(pt) for pt in points]
transform_weights = weightVector(transformed_points, outputs_noise)
est_func = getNonLinearFunc(transform_weights[0], transform_weights[1] , transform_weights[2], transform_weights[3],
                            transform_weights[4], transform_weights[5])
# running it a few times and tallying which function agrees with out 'est_func' to see which ones agrees the most times
num_matches = [0] * len(functions)
for _ in range(25):
    random_point = transformed_points[randint(0, len(transformed_points) - 1)]
    est_value = est_func(random_point)
    for count, func in enumerate(functions):
        if func(random_point) == est_value:
            num_matches[count] += 1
print(num_matches)

#question 10 in hw2
N = 1000
e_out = []
for _ in range(1000):
    out_points = genPoints(-1, 1, -1, 1, N)
    outputs = [targetFunc(pt[1], pt[2]) for pt in out_points]
    outputs_noise = applyNoise(.1, outputs)
    transformed_out_points = [nonLinearTransform(pt) for pt in out_points]
    error = sum([not est_func(transformed_pt)==outputs_noise[count] for count, transformed_pt in enumerate(transformed_out_points)]) / N
    e_out.append(error)
print(np.mean(e_out))


