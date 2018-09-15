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
y_expected = np.array([[0],   [1],   [1],   [0]])

# Model is computationally simple, so a large number of epochs isn't too costly
epochs = 50000

# Input is two binary variables; input space is all possible combinations of
# two binary variables, or X (defined above)
input_size = 2
# Size of the single hidden layer; article cited above
hidden_size = 3
# Output is one binary variable
output_size = 1

# Step size in gradient descent
Learning_rate = 0.1

# Initialize weight matrices between each connected layer with random values
# W1 is weights between input layer and hidden layer
W1 = np.random.uniform(size=(input_size, hidden_size))
# W2 is weights between hidden layer and output layer
W2 = np.random.uniform(size=(hidden_size, output_size))

# Activation values of hidden layer; perhaps analogous to a part of a thought
# process in a literal brain
A1 = sigmoid(np.dot(X, W1))
# Activation values of output layer, i.e. what the model produces
y_produced = sigmoid(np.dot(A1, W2))
