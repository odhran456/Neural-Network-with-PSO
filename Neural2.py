import numpy as np


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


class NeuralNetwork:

    def __init__(self, *args):  # e.g.: NN(3,4,5,6,2)
        self.hidden_layer = list()
        self.weights = list()
        for n in range(len(args)):
            if n == 0:
                self.input_layer = Layer(args[n], "identity")
            elif n == len(args) - 1:
                self.output_layer = Layer(args[n], "identity")
            else:
                self.hidden_layer.append(Layer(args[n], "identity"))

    def assign_weights(self):
        prev_layer = self.input_layer
        for layer in self.hidden_layer:
            self.weights.append(np.random.rand(layer.number_of_nodes, prev_layer.number_of_nodes))
            prev_layer = layer
        self.weights.append(np.random.rand(self.output_layer.number_of_nodes, prev_layer.number_of_nodes))

    def feed_forward(self):
        inputs = np.array([1, 1, 1])
        
        print(self.weights)
        
        # This makes the column vector of node values for the current layer. It takes the initial layer (inputs)
        # and does the dot product with the first array of weights and that gives the next layer on node values
        
        currentMatrix = self.hidden_layer[0].activate_neuron(0, np.dot(self.weights[0], inputs))
        
        # It's put into a numpy array as in the list it had a null column (shape (n, ) ), which is bad,
        # so remake into a numpy nd-array and reshape it into a column vector
        tempMaptrix = np.array([currentMatrix]).transpose()

        prev_matrix = tempMaptrix
        # print(prev_matrix)

        #for loop - check which weight matrix to multiply with

        if self.hidden_layer.__len__() > 1:
            for i in range(1, self.weights.__len__() - 1):
                tempMaptrix = self.hidden_layer[i].activate_neuron(0, np.dot(self.weights[i], prev_matrix))
                prev_matrix = tempMaptrix

        output_nodes = self.output_layer.activate_neuron(0, np.dot(self.weights[-1], tempMaptrix))
        
        return output_nodes
        



    def __repr__(self):
        return "Input Layer: " + str(self.input_layer) + "\n" + "Hidden layers: " + \
               str(self.hidden_layer) + "\n" + "Output Layers: " + str(self.output_layer)


nn = NeuralNetwork(3, 2, 2, 2)

nn.assign_weights()

nn.feed_forward()
