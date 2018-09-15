import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import math

# Goal : build a NN that can model the XOR gate

# Using this website as a guide
# https://www.analyticsindiamag.com/beginners-guide-neural-network-math-python/

# Activation function
def sigmoid(x):
  return 1 / (1 + math.e ** (-1 * x))

# Derivative of activation function
def sigmoid_derivative(x):
  return (sigmoid(x) * (1 - sigmoid(x)))

# Inputs to the XOR gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Desired outputs of XOR gate for each input in X
y = np.array([[0],   [1],   [1],   [0]])
