import cv2
import numpy as np

import gzip
import pickle



#from random import randint, uniform

# Introduction to Neural Networks with OpenCV

# this chapter introduces a family of machine learning model called
# artificial neural networks 
# a key characteristic of these models is that they attempt to learn 
# relationships among variables in a multi-layered fashion.
# 
# they learn multiple functions to predict intermediate results
# before combining these into a single function and use it to predict
# something meaningful, like a class of an object for example
# 
# openCV now have many implementations with neural networks
# and, in particular, neural networks with many layers
# called deep neural networks
# 
# But, we saw in the previous chapters that we can detect object without 
# the use of neural networks, so why do we need to use it in computer 
# vision
# 
# well, neural networks try to provide superior acurracy in the following
# circumstances 
#   there are many inputs variables that are very complex and 
#       and have a nonlinear relationship to each other
#   there are many output variables that are very complex and 
#       have a nonlinear relatinships with the input variables
#       typically, the output variable in a classification are the
#       confidence score for the class, so if there are many classes
#       there are many output variables
#   there are many hidden (unspecified) variables that are very complex
#       and have a nonlinear relationshio with the input and output 
#       variables, so neural networks even aim to model multiples layers
#       of hidden variables that have relationship with each other 
#       and not necessarily with the output or input variables
# these circumstances exist in many, perhaps most, real world problems
# so the promissed advantages of neural networks are enticing
# 
# Understanding ANNs
# 
# let's define a neural networks in terms of its role and components
# first of all, a neural network is a statistical model
#   that is, a pair of elements, namely the space S (a set of observartions)
#   and the probability, P, where P is a distribution that aproximates S
#   in other worlds, a function that would generates a set of observations
#   that is very similar to S.
# 
# we can think of P in two different ways
#   P is a simplification of a complex scenario
#   P is a function the generates S, or at the very least a set of 
#       observations very similar to S
# thus, a artificial neural network is a mode that takes a complex 
# reality, simplify it, and deduce a function (approximately)
# to represent the statiscal observations we would expect from that 
# reality, in a mathematical form
# 
# neural networks, like the other types of machine learning can learn
# from observations in one of the following ways
# Supervised learning
#   under this approach, we want to we want the model's training process
#   to map a known set of input variables to a known set of output 
#   variables. we know the nature of the prediction problem and we delagate
#   the process of finding a function that solves the problem to the 
#   neural network. To train the model, we must provide a set of input 
#   samples along with the correct corresponding output
# unsupervised learning
#   Under this approach a set of output variables aren't known a priori
#   The model's training process must yield a set of output variables
#   as well as a function that maps the input variables to these output
#   variables. for a classification problem, unsupervised learning can 
#   lead to the discovery of a new class. unsupervised learning may
#   use techniques like clustering, but it's not limited to it.
# reinforcement learning
#   this approach turns the typical prediction problem upside down
#   before training the model, before training the model we already have
#   a function that yields values for a known set of output variables 
#   when we feed it with a set of input variables
#   however we might not know the real function that maps the inputs
#   to outputs
#   thus we want a model to produce that predicts a next-in-sequence 
#   optimal inputs, based on the last outputs 
#   During training the model learns from 
#   the score that eventually arises from its actions
#   Essentially, the model must learn to become
#   a good decision maker within the context of a particular system of rewards and punishments
# 
# what if the function that generated the data set is likely
# to take a large number of inputs, an unknown number of inputs
# the strategy that the neural networks use is to delegate work
# to a number of neurons, node, or units, each of which is capable of
# approximating the function that created the input
# the differecen between the approximate function's output
# and the original function's output is called the error
# 
# understanding neurons and perceptrons
# 
# often, to solve a classification problem an nn is designed as a multi-layer perceptron
# in which each neuron acts as a kind of binary classifier called a perceptron
# to put it simply, a perceptron is a function that takes a number of inputs and produces
# a single values
# each of the inputs has an associated weights that signifies its importance in an activation funcition
# the activation function should have a non linear response for example a sigmoid function
# a threshold function, called a discriminant, is applied to the activation function's output
# to convert it into a binary classification
# neurons are interconnected, insofar as one neuron's output can ve an input for many other
# neurons. Each weight defines the strength of the connection between two neurons
# these weights are adaptatives and changes when the learning algorithm is running
# 
# Understanding the layers of a neural network
# 
# there are at leat three layers in a neural network
# the input layer, the hidden layer and the output layer
# a neural network with multiple hidden layers is called
# a deep neural network 
# 
# choosing the size of the input layer
# 
# the number of nodes in the input layer is, by definition, the number
# of inputs into the network.
# 
# choosing the size of the output layer
# 
# for a classifier the number of nodes in the output layer is, by definition,
# the number of classes the network can distinguish
# 
# choosing the size of the hidden layer
# 
# there are no agreed-upon rules of thumb for choosing the size of the hidden layers
# it must be chosen based on experimentation
# 
# Training a basic ANN in OpenCV
# 
# OpenCv provides a class ml_ANN_MLP
# that implements an ann as a multi-layer perceptron
# basic example to see how to work with this
# 

# untrained ANN
""" ann = cv2.ml.ANN_MLP_create()

ann.setLayerSizes(np.array([9,15,9], np.uint8))

ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))

training_samples = np.array( [[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], np.float32)

layout = cv2.ml.ROW_SAMPLE

training_responses = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], np.float32)

data = cv2.ml.TrainData_create(training_samples, layout, training_responses)

ann.train(data)

test_samples = np.array(
    [[1.4, 1.5, 1.2, 2.0, 2.5, 2.8, 3.0, 3.1, 3.8]], np.float32)
prediction = ann.predict(test_samples)
print(prediction)
 """
# Training an ANN classifier in multiple epochs

# Let's create an ANN that attempts to classify animals based on three measurements: weight,
# length, and number of teeth. This is, of course, a mock scenario. Realistically, no one would
# describe an animal with just these three statistics. However, our intent is to improve our
# understanding of ANNs before we start applying them to image data.

""" animals_net = cv2.ml.ANN_MLP_create()
animals_net.setLayerSizes(np.array([3, 50, 4]))
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6,
1.0)
animals_net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
animals_net.setTermCriteria(
    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))
 """
"""Input arrays
weight, length, teeth
"""
"""Output arrays
dog, condor, dolphin, dragon
"""
""" def dog_sample():
    return [uniform(10.0, 20.0), uniform(1.0, 1.5),randint(38, 42)]

def dog_class():
    return [1, 0, 0, 0]

def condor_sample():
    return [uniform(3.0, 10.0), randint(3.0, 5.0), 0]

def condor_class():
    return [0, 1, 0, 0]

def dolphin_sample():
    return [uniform(30.0, 190.0), uniform(5.0, 15.0), randint(80, 100)]

def dolphin_class():
    return [0, 0, 1, 0]

def dragon_sample():
    return [uniform(1200.0, 1800.0), uniform(30.0, 40.0),randint(160, 180)]

def dragon_class():
    return [0, 0, 0, 1]

def record(sample, classification):
    return (np.array([sample], np.float32), np.array([classification], np.float32))

RECORDS = 20000
records = []
for x in range(0, RECORDS):
    records.append(record(dog_sample(), dog_class()))
    records.append(record(condor_sample(), condor_class()))
    records.append(record(dolphin_sample(), dolphin_class()))
    records.append(record(dragon_sample(), dragon_class()))

EPOCHS = 10
for e in range(0, EPOCHS):
    print("epoch: %d" % e)
    for t, c in records:
        data = cv2.ml.TrainData_create(t, cv2.ml.ROW_SAMPLE, c)
        if animals_net.isTrained():
            animals_net.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS |
                cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
        else:
            animals_net.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE |
                cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            
TESTS = 100
dog_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(
        np.array([dog_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 0:
        dog_results += 1

condor_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(
        np.array([condor_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 1:
        condor_results += 1

dolphin_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(
        np.array([dolphin_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 2:
        dolphin_results += 1

dragon_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(
        np.array([dragon_sample()], np.float32))[0])
    print("class: %d" % clas)
    if clas == 3:
        dragon_results += 1

print("dog accuracy: %.2f%%" % (100.0 * dog_results / TESTS))
print("condor accuracy: %.2f%%" % (100.0 * condor_results / TESTS))
print("dolphin accuracy: %.2f%%" % \
(100.0 * dolphin_results / TESTS))
print("dragon accuracy: %.2f%%" % (100.0 * dragon_results / TESTS))
 """
# Recognizing handwritten digits with an ANN

# a handwritten digit is any of the 10 arabic numerals
# written manually with a pen or pencial
# because of it, the appearance of handwritten digits
# can vary significantly
# so almost anybody can write exactly the same way the digits
# this variability makes the problem of recognizing hand written
# digits a non-trivial problem for machine learning
# we will approach this challenge in the following 
# manner:
#   1 load data from a python friendly version of the MNIST database
#       this is a widely used database containing images of handwritten digits
#   2 using the MNIST data, train an ANN in multiples epochs
#   3 Load an image of a sheer of paper with many handwritten digits on it
#   4 based on contour analysis, detect the individual digits on the paper
#   5 Use our ANN to classify the detected digits
#   6 Review the results in order to determine the accuracy of our detector and our
#       ann-based classifier
#   

def load_data():
    mnist = gzip.open('mnist.pkl.gz', 'rb')
    training_data, test_data = pickle.load(mnist)
    mnist.close()
    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((10,), np.float32)
    e[j] = 1.0
    return e

def wrap_data():
    tr_d, te_d = load_data()
    training_inputs = tr_d[0]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_data = zip(te_d[0], te_d[1])
    return (training_data, test_data)

def create_ann(hidden_nodes=60):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784, hidden_nodes, 10]))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
    ann.setTermCriteria(
    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
    100, 1.0))
    return ann

def train(ann, samples=50000, epochs=10):
    tr, test = wrap_data()
    # Convert iterator to list so that we can iterate multiple
    # times in multiple epochs.
    tr = list(tr)

    for epoch in range(epochs):
        print("Completed %d/%d epochs" % (epoch, epochs))
        counter = 0
        for img in tr:
            if (counter > samples):
                break
            if (counter % 1000 == 0):
                print("Epoch %d: Trained on %d/%d samples" % \
                    (epoch, counter, samples))
            counter += 1
            sample, response = img
            data = cv2.ml.TrainData_create(
                np.array([sample], dtype=np.float32),
                cv2.ml.ROW_SAMPLE,
                np.array([response], dtype=np.float32))
            if ann.isTrained():
                ann.train(data, cv2.ml.ANN_MLP_UPDATE_WEIGHTS |
                cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
            else:
                ann.train(data, cv2.ml.ANN_MLP_NO_INPUT_SCALE |
                    cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
    print("Completed all epochs!")

    return ann, test

def predict(ann, sample):
    if sample.shape != (784,):
        if sample.shape != (28, 28):
            sample = cv2.resize(sample, (28, 28),
                interpolation=cv2.INTER_LINEAR)
        sample = sample.reshape(784,)
    return ann.predict(np.array([sample], dtype=np.float32))

def test(ann, test_data):
    num_tests = 0
    num_correct = 0
    for img in test_data:
        num_tests += 1
        sample, correct_digit_class = img
        digit_class = predict(ann, sample)[0]
        if digit_class == correct_digit_class:
            num_correct += 1
    print('Accuracy: %.2f%%' % (100.0 * num_correct / num_tests))

