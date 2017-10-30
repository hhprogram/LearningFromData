import numpy as np
from random import randint
import matplotlib.pyplot as plt


head = 1
trials = []
num_flips = 10
progress = 0
for j in range(100000):
    coins = []
    min_headcount = num_flips
    min_headcount_index = 0
    for i in range(1000):
        heads = 0
        for _ in range(num_flips):
            flip = randint(0,1)
            if flip == head:
                heads +=1
        if heads < min_headcount:
            min_headcount_index = i
            min_headcount = heads
        coins.append(heads)
    trials.append((coins[0] / num_flips, coins[randint(0,999)] / num_flips, coins[min_headcount_index] / num_flips))
    progress = j / 100000 * 100
    if progress % 10 == 0:
        print(progress)

min_headcounts = [trial[2] for trial in trials]
print(np.mean(min_headcounts))
plt.hist(min_headcounts, 10)
plt.show()