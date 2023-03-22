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
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))
        
    
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuples, lists or ndarrays
        """
        # make sure that the vectors have the right shape
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        target_vector = np.array(target_vector).reshape(target_vector.size, 1)

        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector_network = activation_function(self.weights_hidden_out @ output_vector_hidden)
        
        output_error = target_vector - output_vector_network
        tmp = output_error * output_vector_network * (1.0 - output_vector_network)    
        self.weights_hidden_out += self.learning_rate  * (tmp @ output_vector_hidden.T)

        # calculate hidden errors:
        hidden_errors = self.weights_hidden_out.T @ output_error
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * (tmp @ input_vector.T)  
    
    def run(self, input_vector):
        """
        running the network with an input vector 'input_vector'. 
        'input_vector' can be tuple, list or ndarray
        """
        # make sure that input_vector is a column vector:
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size, 1)
        input4hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector_network = activation_function(self.weights_hidden_out @ input4hidden)
        return output_vector_network
            
    def evaluate(self, data, labels):
        """
        Counts how often the actual result corresponds to the
        target result. 
        A result is considered to be correct, if the index of
        the maximal value corresponds to the index with the "1"
        in the one-hot representation,
        e.g.
        res = [0.1, 0.132, 0.875]
        labels[i] = [0, 0, 1]
        """
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
                               learning_rate=0.3)


# """
# XOR
# 0   xor   0 = 0
# 0   xor   1 = 1
# 1   xor   0 = 1
# 1   xor   1 = 0
# """

for i in range(0, 100000):
    simple_network.train([(0, 0)], np.array([(0)]))
    simple_network.train([(0, 1)], np.array([(1)]))
    simple_network.train([(1, 0)], np.array([(1)]))
    simple_network.train([(1, 1)], np.array([(0)]))


print("0? ", simple_network.run([(0, 0)]))
print("1? ", simple_network.run([(0, 1)]))
print("1? ", simple_network.run([(1, 0)]))
print("0? ", simple_network.run([(1, 1)]))

##################
#La prova del calcolo della somma di due digit (da 0 a 3)
#input es: 3+3 => 2 bit + 2bit = 4 nodi
#output 3+3=6 max 0> 4 bit

# simple_network = NeuralNetwork(no_of_in_nodes=4, 
#                                no_of_out_nodes=3, 
#                                no_of_hidden_nodes=16,
#                                learning_rate=0.3)
# for i in range(0, 100000):
#     simple_network.train([(0, 0, 0, 0)], np.array([(0, 0, 0)]))
#     simple_network.train([(0, 0, 0, 1)], np.array([(0, 0, 1)]))
#     simple_network.train([(0, 0, 1, 0)], np.array([(0, 1, 0)]))
#     simple_network.train([(0, 0, 1, 1)], np.array([(0, 1, 1)]))
#     # simple_network.train([(0, 1, 0, 0)], np.array([(0, 0, 1)]))
#     simple_network.train([(0, 1, 0, 1)], np.array([(0, 1, 0)]))
#     simple_network.train([(0, 1, 1, 0)], np.array([(0, 1, 1)]))
#     # simple_network.train([(0, 1, 1, 1)], np.array([(1, 0, 0)]))
#     simple_network.train([(1, 0, 0, 0)], np.array([(0, 1, 0)]))
#     simple_network.train([(1, 0, 0, 1)], np.array([(0, 1, 1)]))
#     simple_network.train([(1, 0, 1, 0)], np.array([(1, 0, 0)]))
#     simple_network.train([(1, 0, 1, 1)], np.array([(1, 0, 1)]))
#     # simple_network.train([(1, 1, 0, 0)], np.array([(0, 1, 1)]))
#     simple_network.train([(1, 1, 0, 1)], np.array([(1, 0, 0)]))
#     simple_network.train([(1, 1, 1, 0)], np.array([(1, 0, 1)]))
#     simple_network.train([(1, 1, 1, 1)], np.array([(1, 1, 0)]))

# # print(simple_network.run([(0, 1, 1, 0)]))
# # print(simple_network.run([(1, 0, 0, 1)]))
# # print(simple_network.run([(1, 0, 1, 1)]))
# # print(simple_network.run([(0, 0, 0, 1)]))

# print(simple_network.run([(0, 1, 0, 0)]))
# print(simple_network.run([(0, 1, 1, 1)]))
# print(simple_network.run([(1, 1, 0, 0)]))


#Se volessi applicare la mia ANN (artificial neural network) al problema delle IRIS
#cosa dovrei fare?
#
# 1) definizione dell'input
# 2) definizione dell'output
# non mi serve altro poichè non mi servono strategie particolari, ci pensa la ANN
# a definire un modello
# iris, cosa era l'input?
# erano 4 valori (sepal len, sepal wid, peta len, petal wid)
# come posso convertire 4 valori in un vettore di bit (input della ANN)
# Inoltre, come posso convertire l'output (le classi) della iris in un vettore di bit?
# Iris-Setosa, Iris-Versicolour, Iris-Virginica => un vettore di tre bit
# 100 setosa
# 010 versicolor
# 001 virginica
# per l'input potrei convertire in millimetri e quindi ottenere
# 4 numeri da 0 a 255, da 0 a 25 cm
# e quindi l'input è un vettore di 4*8 = 32 bit
