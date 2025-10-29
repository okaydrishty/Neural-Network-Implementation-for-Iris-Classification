# Neural Network Implementation for Iris Classification

This project implements a multi-layer neural network from scratch to classify the famous Iris dataset. The network is built using NumPy and demonstrates fundamental concepts of deep learning including forward propagation, backpropagation, and various activation functions.

## Network Architecture

The neural network consists of:
- Input layer: 4 features (iris measurements)
- First hidden layer: 32 neurons with ReLU activation
- Second hidden layer: 16 neurons with ReLU activation
- Output layer: 3 neurons with Softmax activation (for 3 iris classes)

## Features

- Custom implementation of:
  - ReLU activation and its derivative
  - Softmax activation for output layer
  - Forward propagation
  - Backward propagation
  - Cross-entropy loss computation
  - Accuracy calculation
  - Mini-batch gradient descent

## Training Parameters

- Epochs: 50
- Learning rate: 0.05
- Batch size: 32
- Train/Test split: 80/20
- Random seed: 42

## Dependencies

- NumPy: For numerical computations
- Pandas: For data handling
- Matplotlib: For plotting (if visualization is needed)
- Scikit-learn: For loading the Iris dataset and train-test splitting

## Usage

The notebook `iris.ipynb` contains the complete implementation. The network:
1. Loads and preprocesses the Iris dataset
2. Initializes the neural network parameters
3. Trains the model using mini-batch gradient descent
4. Prints training progress every 5 epochs
5. Provides a prediction function for making new classifications

## Implementation Details

The implementation includes several key components:
- Weight initialization with small random values (scaled by 0.01)
- Bias initialization with zeros
- ReLU activation for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Gradient computation and parameter updates
- Mini-batch training with shuffled data

## Performance

The model achieves good accuracy on the Iris classification task through the training process, with progress monitored every 5 epochs showing both loss and accuracy metrics.