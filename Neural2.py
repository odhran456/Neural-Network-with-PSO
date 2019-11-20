import numpy as np
import random


class AFs:

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def step(x):
        return 0 if x < 0 else 1

    @staticmethod
    def identity(x):
        return x


class Neuron:

    def __init__(self, af_name):
        self.af_name = af_name

        if af_name == "identity":
            self.activation_fn = AFs.identity
        elif af_name == "step":
            self.activation_fn = AFs.step
        elif af_name == "sigmoid":
            self.activation_fn = AFs.sigmoid

    def get_af(self, dot_prod):
        return self.activation_fn(dot_prod)

    def change_af(self, new_af_name):
        if new_af_name == "identity":
            self.activation_fn = AFs.identity
            self.af_name = "identity"
        elif new_af_name == "step":
            self.activation_fn = AFs.step
            self.af_name = "step"
        elif new_af_name == "sigmoid":
            self.activation_fn = AFs.sigmoid
            self.af_name = "sigmoid"

    def __repr__(self):
        return self.af_name


class Layer:

    def __init__(self, num_nodes, afname):
        self.number_of_nodes = num_nodes
        self.neuron_list = [Neuron(afname) for neuron in range(num_nodes)]

    #def __repr__(self):
        # return "Layer with " + str(self.number_of_nodes) + " neurons: " + str(self.neuron_list)

    def activate_neuron(self, index, dp):
        # print(self.neuron_list[index])
        return self.neuron_list[index].get_af(dp)

    def change_activation_fn(self, index, new_fn):
        self.neuron_list[index].change_af(new_fn)

class Particle:
    def __init__(self):
        self.position = []
        self.velocity = []
        self.personalBest = 100

    position = []
    velocity = []
    personalBest = 100



class NeuralNetwork:

    def __init__(self, *args):  # e.g.: NN(3,4,5,6,2)
        self.hidden_layer = list()
        self.weights = list()
        self.new_weights_attempt = list()
        # GLOBAL VARIABLE
        self.length_weight_matrix=0
        for n in range(len(args)):
            if n == 0:
                self.input_layer = Layer(args[n], "identity")
            elif n == len(args) - 1:
                self.output_layer = Layer(args[n], "sigmoid")
            else:
                self.hidden_layer.append(Layer(args[n], "sigmoid"))

    def assign_weights(self):
        prev_layer = self.input_layer
        for layer in self.hidden_layer:
            self.weights.append(np.random.rand(layer.number_of_nodes, prev_layer.number_of_nodes))
            prev_layer = layer
        self.weights.append(np.random.rand(self.output_layer.number_of_nodes, prev_layer.number_of_nodes))
        #print(self.weights)


    def assign_weights_from_pso(self, particle):
        # are the rows & columns being reshpaed in the right way ?? ??? ?
        self.new_weights_attempt = []
        self.new_weights_attempt.append(np.array([particle[0].position, particle[1].position, particle[2].position]).reshape(3,1))
        self.new_weights_attempt.append(np.array([particle[3].position, particle[4].position, particle[5].position]).reshape(1,3))
        # print(self.new_weights_attempt[0].shape)
        self.weights = self.new_weights_attempt

        self.length_weight_matrix=0
        for i in range(0, self.weights.__len__()):
            self.length_weight_matrix = self.length_weight_matrix + self.weights[i].shape[0]*self.weights[i].shape[1]


        # flattening_weight = [num for weight in self.weights for num in weight]
        # weight_flattened = np.array([flattening_weight]).flatten()

        return self.weights


    def feed_forward(self):
        inputs = np.array([0.8])
        actual_output = np.array([.71])

        # print(self.weights)

        # This makes the column vector of node values for the current layer. It takes the initial layer (inputs)
        # and does the dot product with the first array of weights and that gives the next layer on node values

        currentMatrix = self.hidden_layer[0].activate_neuron(0, np.dot(self.weights[0], inputs))

        # It's put into a numpy array as in the list it had a null column (shape (n, ) ), which is bad,
        # so remake into a numpy nd-array and reshape it into a column vector
        tempMatrix = np.array([currentMatrix]).transpose()

        prev_matrix = tempMatrix
        # print(prev_matrix)

        # for loop - check which weight matrix to multiply with

        if self.hidden_layer.__len__() > 1:
            for i in range(1, self.weights.__len__() - 1):
                tempMatrix = self.hidden_layer[i].activate_neuron(0, np.dot(self.weights[i], prev_matrix))
                prev_matrix = tempMatrix

        estimated_output = self.output_layer.activate_neuron(0, np.dot(self.weights[-1], tempMatrix))

        #print(estimated_output)

        # calc MSE -> sqrt( (sum of squares) / (num elements)
        sum = 0
        for i in range(0, estimated_output.shape[0]):
            sum = sum + ((actual_output[i] - estimated_output[i])**2)

        mse = np.sqrt(sum / estimated_output.shape[0])

        # This is a numpy array lol
        # print(mse)

        return mse[0]

    # So i think the position matrix is just the weight matrix. Then, each particle (insect) in the swarm
    # has its own neural network, so its own weights matrix, position, velocity matrix etc.


    def PSO(self, numOfParticles):

        insects = 15
        alpha = 0.8
        beta = 2
        gamma = 2
        delta = 0.1
        stepsize = 1
        best = 100

        particles = [Particle() for n in range(numOfParticles)]


        # Make a flat numpy array of the weight matrices in an array called positions
        # flattening_weight = [num for weight in self.weights for num in weight]
        # weight_flattened = np.array([flattening_weight]).flatten()

        # flattening_weight = [num for weight in self.weights for num in weight]
        # weight_flattened = np.array([flattening_weight]).flatten()


        velocities = []
        #for i in range(0, weight_flattened.__len__()):
            #velocities.append(round(random.uniform(-1, 1), 3))


        for insect in range(0, insects - 1):
            # change pos and vel for each particle to random

            # why do we create random weights at the start of feed forward only to never use them and then create random
            # positions which we put into the nn as weights?? Is it just to test the feed fwd?

            #TODO: This should use self.length_weight_matrix and not the number of insects!!!
            for particle in range(0,particles.__len__()):
                particles[particle].position = round(random.uniform(-1, 1), 3)
                particles[particle].velocity = round(random.uniform(-1, 1), 3)

            # print(particles[0].position)
            self.assign_weights_from_pso(particles)


            # Get the mean square error for each of these particle
            a = self.feed_forward()

            # print('--------------------')
            # print(a)
            # print(particle)
            # print(particles[particle].personalBest)
            #record personal best of particle
            if(a < particles[particle].personalBest):
                particles[particle].personalBest = a

            #compare mse with current best
            if( a< best):
                best = a
            print(best)


        # informant_decider = [ np.random.rand(1, numOfParticles) ]
        # print(informant_decider)


    def __repr__(self):
        return "Input Layer: " + str(self.input_layer) + "\n" + "Hidden layers: " + \
               str(self.hidden_layer) + "\n" + "Output Layers: " + str(self.output_layer)


nn = NeuralNetwork(1, 3, 1)

nn.assign_weights()

nn.feed_forward()

nn.PSO(15)

#nn.assign_weights_from_pso()
