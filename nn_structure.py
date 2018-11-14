import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Using Softmax - cross-entropy, we need another parameter: num_example

LEARNING_RATE_SIGMOID = 0.5
LEARNING_RATE_SOFTMAX = 1
REGULARIZATION_STRENGTH_SOFTMAX = 0.001

# Main concept: Neural network is a series of actions to tune the weights of connections
#     First, initialize the weights
#     Second, tune the weights by back-propagation depending on the predicted results
class NeuralNetwork:

    def test(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs, print_status = False, print_hidden_status = False)
        self.prop_backward_softmax(training_outputs)
        self.feed_forward(training_inputs, print_status = False, print_hidden_status = False)

        #print('expected:', np.array(training_outputs).argmax())
        #print('predicted:', np.array(self.output_layer.outputs).argmax())
        #print('-------------------------------------------------------------')
        #print('')

        return(np.array(training_outputs).argmax() == np.array(self.output_layer.outputs).argmax())

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None, num_example = None, lrate = LEARNING_RATE_SOFTMAX):
        self.num_inputs = num_inputs
        
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias, a_func = 'Sigmoid')
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias, a_func = 'Softmax', n_example = num_example)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

        self.n_example = num_example

        self.learning_rate = lrate
        self.regularization_strength = REGULARIZATION_STRENGTH_SOFTMAX
        
# ##########################################################
# #################### Public functions ####################
# ##########################################################

    # We could require the outputs of neurons inside this layer
    def predict(self, test_inputs, print_status = False):
        if print_status: 
            print('[predict]')
            print('tested inputs:', test_inputs)
        self.feed_forward(test_inputs, print_status = print_status, print_hidden_status = False)
        if print_status: print('')

    def train(self, training_inputs, training_outputs, print_status = False):
        if print_status: print('[training]')
        self.feed_forward(training_inputs, print_status = print_status)
        if print_status: print('')
        if print_status: print('[back-propagating]')
        self.prop_backward(training_outputs)

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].get_error(training_outputs[o])
        return total_error

# ##########################################################
# ################### Trivial functions ####################
# ##########################################################

    # Update the outputs of neurons, outputs will be stored inside this layer
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1


# Feed-forward
    def feed_forward(self, inputs, print_status = False, print_hidden_status = False):
        if print_status: print('give input: ', inputs)
        self.hidden_layer.feed_forward(inputs) 
        if (print_status & print_hidden_status): print('outputs from hidden layer: ', self.hidden_layer.outputs)
        self.output_layer.feed_forward(self.hidden_layer.outputs) 
        if print_status: print('predicted outputs: ', self.output_layer.outputs)

# Back-propagation
    # input: the expected outputs of training inputs
    # 1. calculate output layer delta
    # 2. calculate hidden layer delta
    # 3. calculate output layer weight changes
    # 4. calculate hidden layer weight changes
    # 
    # NOTE: delta = partial_d(loss_func) * partial_d(activating_func)
    # 
    # 
    # 
    # TODO: replace the loss function and activating function parts
    def prop_backward_softmax(self, training_outputs):

        # Back propagate from output to hidden layer
        grad_output_layer = (self.output_layer.outputs - training_outputs)

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = grad_output_layer[o] * self.output_layer.neurons[o].get_pd_total_net_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * (pd_error_wrt_weight + self.regularization_strength * self.output_layer.neurons[o].weights[w_ho])
                self.output_layer.neurons[o].bias -= self.learning_rate * np.sum(grad_output_layer)

        grad_hidden_layer = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += grad_output_layer[o] * self.output_layer.neurons[o].weights[h]

            grad_hidden_layer[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_input()
            
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = grad_hidden_layer[h] * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.learning_rate * pd_error_wrt_weight


    def prop_backward(self, training_outputs):

        # pd_errors_wrt_output_neuron_total_net_input: ∂E/∂z, output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].get_pd_error_wrt_total_net_input(training_outputs[o])

        # pd_errors_wrt_hidden_neuron_total_net_input: ∂E/∂zⱼ, hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_input()

        # learning_rate * pd_error_wrt_weight: Δw = α * ∂Eⱼ/∂wᵢⱼ, output neuron Δws
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].get_pd_total_net_input_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * pd_error_wrt_weight
            #self.output_layer.neurons[o].get_status()

        # learning_rate * pd_error_wrt_weight: Δw = α * ∂Eⱼ/∂wᵢⱼ, hidden neuron Δws
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].get_pd_total_net_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.learning_rate * pd_error_wrt_weight
            #self.hidden_layer.neurons[h].get_status()

class NeuronLayer:
# Initialization    
    # Each layer contains several neurons, a common bias and outputs of neurons
    # Only the neurons and the bias need initialization 
    # The output of neuron need no initialization since it'll be updated later
    def __init__(self, num_neurons, bias, a_func = 'Sigmoid', n_example = None):

        self.outputs = []

        self.bias = bias if bias else random.random()

        self.ativating_func = a_func

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

# ##########################################################
# #################### Public functions ####################
# ##########################################################

    # Update the outputs of neurons, outputs will be stored inside this layer
    def feed_forward(self, inputs):
        self.outputs = []
        neuron_outputs = []
        for neuron in self.neurons:
            neuron_outputs.append(neuron.get_output(inputs))
        if self.ativating_func == 'Sigmoid':
            self.outputs = self.squash(np.array(neuron_outputs))
        elif self.ativating_func == 'Softmax':
            self.outputs = self.relu(self.softmax(neuron_outputs))
        for n_o in range(len(self.neurons)):
            self.neurons[n_o].set_output(self.outputs[n_o])

    def get_weights(self):
        neuron_weights = []
        for neuron in self.neurons:
            neuron_weights = np.append(neuron_weights, neuron.weights)
        return neuron_weights

    def get_bias(self):
        neuron_bias = []
        for neuron in self.neurons:
            neuron_bias.append(neuron.bias)
        return neuron_bias

# ##########################################################
# ################### Trivial functions ####################
# ##########################################################

    def relu(self, total_net_input):
        return np.maximum(0, total_net_input)

    def softmax(self, total_net_input):
        return np.exp(total_net_input) / np.sum(np.exp(total_net_input))

    def squash(self, total_net_input):
        return 1 / (1 + np.exp(-total_net_input))

class Neuron:

# Initialization    
	# Every type of neuron except input neurons need initialization of a bias and weights.
    # However, The weights of neuron could only be initialized after the next layer of neuron constructed.
    # The output of neuron need no initialization since it'll be updated later
    def __init__(self, bias):
        self.bias = bias
        self.output = None
        self.weights = []

# ##########################################################
# #################### Public functions ####################
# ##########################################################

# Activate a neuron
    # Using ReLU
    def get_output(self, inputs):
        self.set_inputs(inputs)
        return self.get_summed_weighted_inputs_with_bias(inputs)

    def get_status(self):
        print('inputs: ', self.inputs)
        print('outputs: ', self.output)
        print('weights: ', self.weights)

# Error getting functions
    def get_pd_error_wrt_total_net_input(self, target_output):
        return self.get_pd_error_wrt_output(target_output) * self.get_pd_total_net_input_wrt_input();

    def get_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def get_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def get_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

    def get_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

# ##########################################################
# ################### Trivial functions ####################
# ##########################################################
    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_output(self, output):
        self.output = output

    def get_summed_weighted_inputs_with_bias(self, inputs):
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        return total + self.bias


if __name__ == "__main__":
	# XOR example test
	n_sample = 101
	training_sets = [
			 [[0, 0], [1, 0]],
			 [[0, 1], [0, 1]],
			 [[1, 0], [0, 1]],
			 [[1, 1], [1, 0]]
	]
	nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]), num_example=n_sample, lrate = 0.8)
	for i in range(n_sample):
		training_inputs, training_outputs = random.choice(training_sets)
		nn.test(training_inputs = training_inputs, training_outputs = training_outputs)


	# CS231n example test
	np.random.seed(0)
	N = 100; D = 2; K = 3
	X = np.zeros((N*K, D)); y = np.zeros((N*K, K), dtype = 'uint8')
	for j in range(K):
		ix = range(N*j, N*(j + 1))
		r = np.linspace(0.0, 1, N)
		t = np.linspace(j*4, (j+1) * 4, N) + np.random.randn(N) * 0.2
		X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
		y[ix, j] = np.ones(len(ix))
	accuracy = []
	n_sample = 1000
	nn = NeuralNetwork(len([X[0, 0], X[0, 1]]), 100, len([y[0, 0], y[0, 1], y[0, 2]]), num_example=300, lrate = 0.8)
	for i in range(n_sample):
		sel_id = np.random.choice(len(X), 1)
		training_inputs = [X[sel_id, 0], X[sel_id, 1]]
		training_outputs = [y[sel_id, 0], y[sel_id, 1], y[sel_id, 2]]
		accuracy.append(nn.test(training_inputs = training_inputs, training_outputs = training_outputs))
	print('Accuracy:', np.mean(accuracy))
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	out_hidden = np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], nn.hidden_layer.get_weights().reshape(100,2).T) + nn.hidden_layer.get_bias())
	Z = np.dot(out_hidden, nn.output_layer.get_weights().reshape(3,100).T) + nn.output_layer.get_bias()
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c = np.argmax(y, axis=1), s = 40, cmap = plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()
