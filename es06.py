import numpy as np
from scipy.stats import truncnorm

def sigma(x):
    return 1 / (1 + np.exp(-x))

def sigma_derivation(x):
    return sigma(x)*(1-sigma(x))

def ReLU(x):
    return np.maximum(0.0, x)

# derivation of relu
def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return 1


activation_function=sigma

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
            
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None):  
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.learning_rate = learning_rate 
        self.bias = bias
        self.create_weight_matrices()
    
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural 
        network with optional bias nodes"""   
        bias_node = 1 if self.bias else 0 
        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                        self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                         self.no_of_hidden_nodes + bias_node))

        
    def train(self, input_vector, target_vector):
        """ input_vector and target_vector can be tuple, list or ndarray """

        # make sure that the vectors have the right shap
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)        
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate( (input_vector, [[self.bias]]) )
        target_vector = np.array(target_vector).reshape(target_vector.size, 1)

        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        if self.bias:
            output_vector_hidden = np.concatenate( (output_vector_hidden, [[self.bias]]) ) 
        output_vector_network = activation_function(self.weights_hidden_out @ output_vector_hidden)
        
        output_error = target_vector - output_vector_network  
        # update the weights:
        tmp = output_error * output_vector_network * (1.0 - output_vector_network)     
        self.weights_hidden_out += self.learning_rate  * (tmp @ output_vector_hidden.T)

        # calculate hidden errors:
        hidden_errors = self.weights_hidden_out.T @ output_error
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = (tmp @input_vector.T)[:-1,:]     # last row cut off,
        else:
            x = tmp @ input_vector.T
        self.weights_in_hidden += self.learning_rate *  x


           
    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'. 
        'input_vector' can be tuple, list or ndarray
        """
        # make sure that input_vector is a column vector:
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate( (input_vector, [[1]]) )
        input4hidden = activation_function(self.weights_in_hidden @ input_vector)
        if self.bias:
            input4hidden = np.concatenate( (input4hidden, [[1]]) )
        output_vector_network = activation_function(self.weights_hidden_out @ input4hidden)
        return output_vector_network
            
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i].argmax():
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=1, 
                               no_of_hidden_nodes=16,
                               learning_rate=0.4,
                               bias=1)


# """
# XOR
# 0   xor   0 = 0
# 0   xor   1 = 1
# 1   xor   0 = 1
# 1   xor   1 = 0
# """

"""
for i in range(0, 1000):
    simple_network.train([(0, 0)], np.array([(0)]))
    simple_network.train([(0, 1)], np.array([(1)]))
    simple_network.train([(1, 0)], np.array([(1)]))
    simple_network.train([(1, 1)], np.array([(0)]))

print(simple_network.run([(0, 0)]))
print(simple_network.run([(0, 1)]))
print(simple_network.run([(1, 0)]))
print(simple_network.run([(1, 1)]))
"""

"""Esempio con una classificazione di fiori artificiali: strange_flowers"""
c = np.loadtxt("data/strange_flowers.txt", delimiter=" ")

data = c[:, :-1]
labels = c[:, -1]
n_classes = int(np.max(labels)) # in our case 1, ... 4
print(data[:5])

# Converte le etichette in codici univoci
labels_one_hot = np.arange(1, n_classes+1) == labels.reshape(labels.size, 1)
labels_one_hot = labels_one_hot.astype(np.float64)
print(labels_one_hot[:3])

from sklearn.model_selection import train_test_split

res = train_test_split(data, labels_one_hot, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_labels, test_labels = res    
print(train_labels[:10])

#La normalizzazione dei dati
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
train_data = scaler.fit_transform(train_data) #  fit and transform
test_data = scaler.transform(test_data) #  transform

simple_network = NeuralNetwork(no_of_in_nodes=4, 
                               no_of_out_nodes=4, 
                               no_of_hidden_nodes=20,
                               learning_rate=0.3)

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])
    
print(simple_network.evaluate(train_data, train_labels))

