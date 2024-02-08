# AIT Budapest - Deep Learning course
## Assignment 1.
## Created by: Bálint Gyires-Tóth

# Introduction

In this assignment, we focus on some concepts of neural networks and backpropagation. Neural networks are a fundamental component of deep learning, and understanding how they work is crucial for building and training effective models. Backpropagation is an algorithm used to train neural networks by adjusting the weights and biases of the network based on the error between the predicted and actual outputs. Through this assignment, we will delve into some details of the theory behind neural networks and backpropagation, and implement them in code to gain hands-on experience.

Please always write your anwser between the "------" lines. 

Let's get started!

## Rules

Your final score will be penalized if you engage in the following behaviors:

1. 20% penalty for not using or following correctly the structure of this file.
2. 20% penalty for late submission within the first 24 hours. Late submissions after the first 24 hours will not be accepted. 
3. 20% penalty for lengthy answers.
3. 40% penalty for for joint work or copying, including making the same, non-tipical mistakes, to all students concerned.

# Theory (50 points)
We have the following neural network architecture:

Input: 10 variables
Layer 1: Fully-connected layer with 16 neurons and sigmoid activation, with bias
Layer 2: Fully-connected layer with 1 neuron and sigmoid activation, no bias

The fully-connected layer is defined as s^(i+1) = a^(i)*W^(i), where 

s^(i): output of the i-th fully-connected layer without activation.
a^(i): output of the activation function of the i-th layer.
a^(1): X (the input data).
W^(i): weight matrix of the i-th layer. 

Use the notation as above. For partial deriavative use the @ sign. The cost function is MSE denoted by C, and the ground truth is y. 

Question 1: Define the number of parameters in the neural network. (10 points)
------
Answer 1: 
------

Question 2: Define the output of Layer 1 after the activation w.r.t. the input data. (10 points)
------
Answer 2:
------

Question 3: Define the output of Layer 2 after the activation w.r.t. the input data. (10 points)
------
Answer 3:
------

Question 4: Define the gradient of W^(2) w.r.t. the loss. (10 points)
------
Answer 4:
------

Question 5: Define the gradient of W^(1) w.r.t. the loss. (10 points)
------
Answer 5:
------

# Practice (50 points)

Please submit your work based on the shared notebook. Always test your solution in the shared notebook before submission. Only modify the specified code snippet. 

Task 1: Compleate the training loop with early stopping method. You can use any existing variable in the code if needed. (25 points) 
------
Answer 6:

        for i in range(epochs):
            # Training phase - sample by sample
            train_err = 0
            for k in range(X_train.shape[0]):
                model.propagate_forward( X_train[k] )
                train_err += model.propagate_backward( Y_train[k], lrate )
            train_err /= X_train.shape[0]

            # Validation phase
            valid_err = 0
            o_valid = np.zeros(X_valid.shape[0])
            for k in range(X_valid.shape[0]):
                o_valid[k] = model.propagate_forward(X_valid[k])
                valid_err += (o_valid[k]-Y_valid[k])**2
            valid_err /= X_valid.shape[0]
------

Task 2: Compleate the backpropagation algorithm with momentum method. You can use any existing variable in the code if needed. (25 points) 
------
Answer 7:

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw    = -lrate*np.dot(layer.T,delta)
            self.weights[i] += dw 
------
