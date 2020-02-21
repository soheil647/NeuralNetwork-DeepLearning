#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from scipy.optimize import fmin_bfgs

class NeuralNetwork:
    """A simple neural network."""

    def __init__(self, hidden_layers=(25,), reg_lambda=0, num_labels=2):
        """Instantiates the class."""
        self.__hidden_layers = tuple(hidden_layers)
        self.__lambda = reg_lambda
        if num_labels > 2:
            self.__num_labels = num_labels
        else:
            self.__num_labels = 1

    def train(self, training_set, iterations=500):
        """Trains itself using the sequence data."""
        if len(training_set) > 2:
            self.__X = np.matrix([example[0] for example in training_set])
            if self.__num_labels == 1:
                self.__y = np.matrix([example[1] for example in training_set]).reshape((-1, 1))
            else:
                eye = np.eye(self.__num_labels)
                self.__y = np.matrix([eye[example[1]] for example in training_set])
        else:
            self.__X = np.matrix(training_set[0])
            if self.__num_labels == 1:
                self.__y = np.matrix(training_set[1]).reshape((-1, 1))
            else:
                eye = np.eye(self.__num_labels)
                self.__y = np.matrix([eye[index] for sublist in training_set[1] for index in sublist])
        self.__m = self.__X.shape[0]
        self.__input_layer_size = self.__X.shape[1]
        self.__sizes = [self.__input_layer_size]
        self.__sizes.extend(self.__hidden_layers)
        self.__sizes.append(self.__num_labels)
        initial_theta = []
        for count in range(len(self.__sizes) - 1):
            epsilon = np.sqrt(6) / np.sqrt(self.__sizes[count]+self.__sizes[count+1])
            initial_theta.append(np.random.rand(self.__sizes[count+1],self.__sizes[count]+1)*2*epsilon-epsilon)
        initial_theta = self.__unroll(initial_theta)
        self.__thetas = self.__roll(fmin_bfgs(self.__cost_function, initial_theta, fprime=self.__cost_grad_function, maxiter=iterations))

    def predict(self, X):
        """Returns predictions of input test cases."""
        return self.__cost(self.__unroll(self.__thetas), 0, np.matrix(X))

    def __cost_function(self, params):
        """Cost function used by fmin_bfgs."""
        return self.__cost(params, 1, self.__X)

    def __cost_grad_function(self, params):
        """Cost gradient used by fmin_bfgs."""
        return self.__cost(params, 2, self.__X)

    def __cost(self, params, phase, X):
        """Computes activation, cost function, and derivative."""
        params = self.__roll(params)
        a = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # This is a1
        calculated_a = [a] # a1 is at index 0, a_n is at index n-1
        calculated_z = [0] # There is no z1, z_n is at index n-1
        for i, theta in enumerate(params): # calculated_a now contains a1, a2, a3 if there was only one hidden layer (two theta matrices)
            z = calculated_a[-1] * theta.transpose() # z_n = a_n-1 * Theta_n-1'
            calculated_z.append(z) # Save the new z_n
            a = self.sigmoid(z) # a_n = sigmoid(z_n)
            if i != len(params) - 1: # Don't append extra ones for the output layer
                a = np.concatenate((np.ones((a.shape[0], 1)), a), axis=1) # Append the extra column of ones for all other layers
            calculated_a.append(a) # Save the new a

        if phase == 0:
            if self.__num_labels > 1:
                return np.argmax(calculated_a[-1], axis=1)
            return np.round(calculated_a[-1])

        J = np.sum(-np.multiply(self.__y, np.log(calculated_a[-1]))-np.multiply(1-self.__y, np.log(1-calculated_a[-1])))/self.__m; # Calculate cost
        if self.__lambda != 0: # If we're using regularization...
            J += np.sum([np.sum(np.power(theta[:,1:], 2)) for theta in params])*self.__lambda/(2.0*self.__m) # ...add it from all theta matrices

        if phase == 1:
            return J

        reversed_d = []
        reversed_theta_grad = []
        for i in range(len(params)): # For once per theta matrix...
            if i == 0: # ...if it's the first one...
                d = calculated_a[-1] - self.__y # ...initialize the error...
            else: # ...otherwise d_n-1 = d_n * Theta_n-1[missing ones] .* sigmoid(z_n-1)
                d = np.multiply(reversed_d[-1]*params[-i][:,1:], self.sigmoid_grad(calculated_z[-1-i])) # With i=1/1 hidden layer we're getting Theta2 at index -1, and z2 at index -2
            reversed_d.append(d)
            theta_grad = reversed_d[-1].transpose() * calculated_a[-i-2] / self.__m
            if self.__lambda != 0:
                theta_grad += np.concatenate((np.zeros((params[-1-i].shape[0], 1)), params[-1-i][:,1:]), axis=1) * self.__lambda / self.__m# regularization
            reversed_theta_grad.append(theta_grad)
        theta_grad = self.__unroll(reversed(reversed_theta_grad))
        return theta_grad

    def __roll(self, unrolled):
        """Converts parameter array back into matrices."""
        rolled = []
        index = 0
        for count in range(len(self.__sizes) - 1):
            in_size = self.__sizes[count]
            out_size = self.__sizes[count+1]
            theta_unrolled = np.matrix(unrolled[index:index+(in_size+1)*out_size])
            theta_rolled = theta_unrolled.reshape((out_size, in_size+1))
            rolled.append(theta_rolled)
            index += (in_size + 1) * out_size
        return rolled

    def __unroll(self, rolled):
        """Converts parameter matrices into an array."""
        return np.array(np.concatenate([matrix.flatten() for matrix in rolled], axis=1)).reshape(-1)

    def sigmoid(self, z):
        """Sigmoid function to emulate neuron activation."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_grad(self, z):
        """Gradient of sigmoid function."""
        return np.multiply(self.sigmoid(z), 1-self.sigmoid(z))

    def grad(self, params, epsilon=0.0001):
        """Used to check gradient estimation through slope approximation."""
        grad = []
        for x in range(len(params)):
            temp = np.copy(params)
            temp[x] += epsilon
            temp2 = np.copy(params)
            temp2[x] -= epsilon
            grad.append((self.__cost_function(temp)-self.__cost_function(temp2))/(2*epsilon))
        return np.array(grad)