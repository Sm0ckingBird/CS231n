#!/usr/bin/env python3
def fibonacci(x):
	n, a, b = 0, 0, 1
	while(n < x):
		yield b
		a, b = b, a+b
		n = n + 1

def triangle():
	N = [1]
	while True:
		yield N
		N.append(0)
		N = [N[i-1]+N[i] for i in range(len(N))]
		
def odd_nums():
	n = 1
	while True:
		n = n + 2
		yield n

def prime_filter(n):
	return lambda x: x % n > 0

def prime_generator():
	nums = odd_nums()
	while True:
		first = next(nums)
		nums = filter(prime_filter(first),nums)
		yield first

def is_palindrome(n):
    string = str(n)
    string = list(string)
    string2 = [string[len(string)-i-1] for i in range(len(string))]
    return string == string2
import numpy as np
from cs231n.data_utils import load_CIFAR10
cifar10_dir = '../assignment1/cs231n/datasets/cifar-10-batches-py'
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    print(y_train == y)



