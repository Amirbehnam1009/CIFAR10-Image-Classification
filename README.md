# CIFAR10 Image Classification Using Fully Connected Neural Network

"In this endeavor, we embarked on constructing a comprehensive neural network from its foundational principles, integrating the CIFAR10 dataset for training purposes. Essential learnings encompassed the intricacies of data preprocessing, forward and backward propagation, and the optimization power of vectorization.

At its core, this project aimed at grasping the pivotal role of Vectorization in expediting the computational efficiency of neural network architectures, thereby illuminating its significance in contemporary machine learning paradigms."

## About

> Under The Supervision of [Prof.Mohammad Mehdi Ebadzadeh](https://scholar.google.com/citations?user=080Y_lUAAAAJ&hl=en)

> Spring 2022

## Libararies

**Matplotlib, Numpy, Scikit-image, PIL (pillow), Glob, os, time**

## Steps
### **1. Data Preprocessing**

#### Step 1: Read and Save Data

- Read the first 4 sets of datasets (airplane, automobile, bird, and cat classes) in both training and test datasets.
- Save data in matrix format `(n_samples, width, height, channels)`.
- Store labels in a one-hot matrix.

#### Steps 2-5: Data Transformation

1. **Convert Images to Grayscale**: Reduce computational complexity by converting images to grayscale.
2. **Normalization**: Scale pixel values to the range of zero to one by dividing by 255.
3. **Flatten Data**: Reshape data to have dimensions `(n_samples, 1024)` to match the input layer of the network.
4. **Shuffle Data**: Randomized the order of data samples before training. The same shuffling order is applied to both data and label matrices.

### **2. Calculating The Output (Feedforward)**

To calculate the output in a neural network, each layer performs matrix/vector multiplication followed by the application of the sigmoid function. Here's a summarized process:

1. **Data Selection**: A subset of 200 data points is chosen from the training set.
2. **Weights and Biases Initialization**: The weights are initialized with random values, and biases are set to zero vectors.
3. **Output Computation**: Using matrix multiplication and the sigmoid function, the output is calculated for the selected data points.
4. **Model Inference**: Each data point is assigned to a category based on the neuron with the highest activation in the last layer.
5. **Accuracy Evaluation**: The model's accuracy is evaluated by comparing its predictions with the actual labels. At this stage, due to random initialization, the expected accuracy is around 25%.

**Note**: NumPy is used for matrix operations.

### **3. Implementation Of Backpropgation**

Backpropagation is employed to iteratively refine the model's parameters, thereby minimizing the discrepancy between predicted and actual outputs. Here's a summary of the process:

- **Parameter Tuning**: Key parameters such as batch size, learning rate, and number of epochs are carefully selected.
- **Implementation**: Utilizing the provided pseudo-code, backpropagation is implemented to update weights and biases iteratively.

#### Performance Evaluation

We assess the model's performance and execution time:

- **Accuracy Evaluation**: The model's accuracy on a subset of 200 data points is reported.
- **Execution Time**: The duration of the learning process is measured.
- **Expected Outcome**: Given the chosen parameters and dataset, an average accuracy of approximately 30% is expected, considering the impact of random initialization.

#### Cost Function Visualization

The reduction in average cost per epoch is visualized:

- **Average Cost Calculation**: The average cost per epoch is computed.
- **Graphical Representation**: A graph is plotted to illustrate the decline in average cost over epochs.

### **4. Vectorization**

Vectorization allows us to perform operations efficiently using matrix operations, significantly reducing execution time.

#### Implementation

- **Feedforward Stage**: We initially implemented the feedforward algorithm in a vectorized form, enhancing computational efficiency.
- **Backpropagation Vectorization**: In this step, we've extended vectorization to the backpropagation process, eliminating the need for iterative loops and further improving performance.

#### Performance Evaluation

- **Increased Epochs**: To evaluate the model comprehensively, we increased the number of epochs to 20.
- **Evaluation Metrics**: Our assessment includes reporting the accuracy of the final model, the execution time of the learning process, and the cost over time.
- **Execution Iterations**: We executed the code multiple times to account for variations in execution speed and presented the average results.

### **5. Testing The Model**

In this step, we evaluate the performance of our optimized model using the complete dataset of 4 classes (8000 data points).


#### Implementation

- **Training Setup**: The model is trained with specific hyperparameters without explicitly instructing readers to replicate the process.
- **Evaluation Metrics**: We analyze the model's accuracy on both the train and test sets to understand its performance.
- **Average Cost Plot**: Visualizing the learning process by plotting the average cost over time.
