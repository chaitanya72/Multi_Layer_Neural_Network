# using tensorflow_version 2.x

import tensorflow as tf
import numpy as np
import math
if tf.executing_eagerly() == False:
	tf.enable_eager_execution()

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' 

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
         """
        
        
        if len(self.weights)==0:
            W = tf.Variable(tf.zeros((self.input_dimension,num_nodes)),trainable=True)
        else:
            #shape_tf = tf.shape(self.weights[-1])
            shape_tf = self.weights[-1].get_shape()
            print(shape_tf)
            W = tf.Variable(tf.zeros((shape_tf[1],num_nodes)),trainable=True)
        b = tf.Variable(tf.zeros((1,num_nodes)),trainable=True) #chaeck this
        self.weights.append(W)
        self.biases.append(b)
        self.activations.append(activation_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        inputs = X.copy()
        for layer in range(0,len(self.weights)):
            outputs = self.activations[layer](tf.matmul(inputs,self.weights[layer])+self.biases[layer])
            inputs = outputs
        return outputs

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
        for epoch in range(0,num_epochs):
            num_batches = math.ceil(X_train.shape[0]/batch_size)
            batch_x = np.array_split(X_train,num_batches,axis=0)
            batch_y = np.array_split(y_train,num_batches)
            #loss_1 = 0
            for batch in range(0,len(batch_x)):
                inputs = batch_x[batch].copy()
                with tf.GradientTape(persistent=True) as tape1:
                    with tf.GradientTape() as tape2:
                        for layer in range(0,len(self.weights)):
                            
                            tape1.watch([self.weights[layer]])
                            tape2.watch([self.biases[layer]])
                            
                            outputs = self.activations[layer](tf.matmul(inputs,self.weights[layer])+ self.biases[layer])
                            inputs = outputs
                            
                            loss_1 = self.loss(batch_y[batch],outputs)
                            
                grad_list = tape1.gradient(loss_1,[self.weights])
                grad_list_bias = tape2.gradient(loss_1,[self.biases])
                
                del tape1
                for layer in range(0,len(self.weights)):
                    
                    self.weights[layer] = self.weights[layer] - alpha*np.array((grad_list[0][layer]))
                    self.biases[layer] = self.biases[layer] - alpha*grad_list_bias[0][layer]
                    
                

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred,axis =1)
        print(y_pred.shape)
        count = 0
        for i in range(0,len(y)):
            if y[i]!=y_pred[i]:
              count = count + 1
        return count/len(y)

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred,axis = 1)
        num_class = self.weights[-1].get_shape()
        print(num_class[1])
        num_classes = 10
        cm = np.zeros((num_class[1],num_class[1]))
        for i in range(0,len(y)):
            cm[y[i]][y_pred[i]] = cm[y[i]][y_pred[i]] + 1
        return cm


if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        #print(b.get_shape())
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))

