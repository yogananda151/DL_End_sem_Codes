# Import numpy library for numerical calculations and array operations
import numpy as np

# Import PyTorch core library for deep learning and tensor computations
import torch

# Import torchvision library containing popular computer vision datasets and model architectures
import torchvision

# Import matplotlib's pyplot module for visualizing and displaying images and charts
import matplotlib.pyplot as plt

# Import time module function to track code execution duration
from time import time

# Import standard computer vision datasets and image transformation pipelines from torchvision
from torchvision import datasets, transforms

# Import neural network layers (nn) and optimization algorithms (optim) from PyTorch
from torch import nn, optim

# Print the installed PyTorch version to console to verify setup
print(torch.__version__)


# Define an image transformation pipeline using transforms.Compose to chain multiple operations
transform = transforms.Compose([
    # Convert input PIL image or numpy array (0-255) into a PyTorch tensor scaled to [0.0, 1.0]
    transforms.ToTensor(),
    # Normalize tensor values: (image - mean) / std, bringing pixels to range [-1, 1]
    transforms.Normalize((0.5,), (0.5,)), # mean of 0.5, standard deviation of 0.5
])

# Download and load the MNIST training dataset (60,000 images of handwritten digits) with applied transformations
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)

# Download and load the MNIST test dataset (10,000 images) with applied transformations
testset = datasets.MNIST('./', download=True, train=False, transform=transform)


# Create DataLoader for training set to automatically batch 64 images together and shuffle order each epoch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Create DataLoader for test set to batch 64 images together for evaluation
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Convert the trainloader into an iterator to access individual batches
dataiter = iter(trainloader)

# Fetch the first mini-batch containing images and their corresponding ground truth labels
images, labels = next(dataiter)


# Print shape of image batch: (batch_size=64, channels=1, height=28, width=28)
print(images.shape)

# Print shape of labels batch: (batch_size=64) containing digit classes 0-9
print(labels.shape)

# Squeeze out channel dimension (28x28) and display the first image in grayscale using matplotlib
plt.imshow(images[0].numpy().squeeze(), cmap='gray')


# Convert testloader into an iterator to inspect validation/test batch
dataiter = iter(testloader)

# Fetch the first batch of test images and labels
images, labels = next(dataiter)

# Print test images batch dimensions
print(images.shape)

# Print test labels batch dimensions
print(labels.shape)


# Define a Convolutional Neural Network class inheriting from PyTorch's nn.Module
class Net(nn.Module):  # it allows your class to use PyTorch's standard model functionality 

    # Constructor method to define and initialize layers
    def __init__(self):
        # Call the parent nn.Module constructor
        super(Net, self).__init__()

        # Group convolutional, batchnorm, activation, and pooling layers sequentially
        self.cnn_layers = nn.Sequential(
            # 1st Conv2D layer: 1 input channel (grayscale), 4 output filters, 3x3 kernel, padding=1 keeps size 28x28
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            # Batch Normalization across 4 feature channels to stabilize training
            nn.BatchNorm2d(4),
            # ReLU activation function to introduce non-linearity (max(0, x))
            nn.ReLU(inplace=True),
            # Max pooling with 2x2 window reduces spatial resolution by half: 28x28 -> 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd Conv2D layer: 4 input channels, 4 output filters, 3x3 kernel, keeps size 14x14
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            # Batch Normalization across 4 feature channels
            nn.BatchNorm2d(4),
            # ReLU activation function
            nn.ReLU(inplace=True),
            # Max pooling reduces spatial resolution by half: 14x14 -> 7x7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected linear layer block for classification
        self.linear_layers = nn.Sequential(
            # Linear layer mapping flattened feature maps (4 channels * 7 * 7 = 196) to 10 digit classes (0-9)
            nn.Linear(4 * 7 * 7, 10)
        )

    # Define the forward pass to describe how input data moves through the layers
    def forward(self, x):
        # Pass input image through convolutional feature extractor
        x = self.cnn_layers(x)
        # Flatten the 4D feature map (batch_size, 4, 7, 7) into 2D tensor (batch_size, 196)
        x = x.view(x.size(0), -1)
        # Pass flattened representations through final linear classification layer
        x = self.linear_layers(x)
        # Return raw class scores (logits)
        return x

# Instantiate the Net model architecture
model = Net()

# Define Adam optimizer with learning rate 0.01 to update model parameters
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define CrossEntropyLoss function which combines LogSoftmax and NLLLoss for multi-class classification
criterion = nn.CrossEntropyLoss()  #This creates the loss function used to measure the difference

# Check if a CUDA-enabled GPU is available on the system
if torch.cuda.is_available():
    # Move model parameters to GPU memory for faster computation
    model = model.cuda()
    # Move loss function criteria to GPU
    criterion = criterion.cuda()

# Print the complete model architecture details to the console
print(model)

# Create a random dummy input tensor with shape (1 sample, 1 channel, 28 height, 28 width)
input = torch.randn(1,1,28,28)

# Check if CUDA is available to move dummy input to GPU
if torch.cuda.is_available():
    input = input.cuda()

# Pass dummy input through model to test forward pass execution
out = model(input)

# Print model output logits for dummy input (shape: 1 x 10)
print(out)


# Import summary utility to display detailed layer-by-layer parameter counts and shapes
from torchsummary import summary

# Print summary table of model showing output shapes and parameter numbers for (1, 28, 28) input
summary(model, input_size=(1, 28, 28))



# Loop through training dataset for 10 complete passes (epochs)
for i in range(10): # This outer loop runs the training process ten times.
    # Initialize accumulator to sum batch losses during the epoch
    running_loss = 0
    # Iterate through each batch of images and labels in the trainloader
    for images, labels in trainloader: # This inner loop goes through all the batches in the training dataset.      

        # Check if GPU is available to transfer batch to GPU
        if torch.cuda.is_available():
            # Move images tensor to GPU
            images = images.cuda()
            # Move labels tensor to GPU
            labels = labels.cuda()

        # Reset gradients of all model parameters to zero before backward pass
        optimizer.zero_grad()
        
        # Forward pass: compute model predictions (logits) for current batch
        output = model(images)

        # Compute cross-entropy loss between model predictions and true class labels
        loss = criterion(output, labels)
        
        # Backward pass: compute gradients of loss with respect to all model weights
        #This is where the model learns by backpropagating
        loss.backward()
        
        # Update model weights based on computed gradients using Adam optimizer
        #And optimizes its weights here
        optimizer.step()
        
        # Add scalar loss of current batch multiplied by batch size or batch loss item
        running_loss += loss.item()
    # Executed after completing all batches in the epoch
    else:
        # Print epoch number and average training loss across all batches
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))


# Initialize counters for correctly predicted images and total tested images
correct_count, all_count = 0, 0

# Loop through all batches in test dataset
for images, labels in testloader:
    # Loop through each image individually in the current test batch
    for i in range(len(labels)):
        # Check if GPU is available
        if torch.cuda.is_available():
            # Move images to GPU
            images = images.cuda()
            # Move labels to GPU
            labels = labels.cuda()

        # Extract i-th image and reshape to (1, 1, 28, 28) for single-image inference
        img = images[i].view(1, 1, 28, 28)

        # Disable gradient computation during evaluation to save memory and computations
        with torch.no_grad():
            # Forward pass: get model prediction logits for this image
            logps = model(img)

        # Apply exponential to convert logits into probabilities
        ps = torch.exp(logps)

        # Move probabilities tensor to CPU and convert to a Python list
        probab = list(ps.cpu()[0])

        # Find the index of the highest probability which represents the predicted class digit
        pred_label = probab.index(max(probab))

        # Get the ground truth label from the labels tensor on CPU
        true_label = labels.cpu()[i]

        # Check if the predicted label matches the ground truth label
        if(true_label == pred_label):
            # Increment correct predictions counter
            correct_count += 1
        # Increment total images tested counter
        all_count += 1


# Print the total count of images evaluated on the test set
print("Number Of Images Tested =", all_count)

# Calculate and print overall test classification accuracy (correct / total)
print("\nModel Accuracy =", (correct_count/all_count))
