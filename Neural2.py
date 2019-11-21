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
        self.personalBestPosition = []
        self.informantIndex = -1
        self.group = -1
        self.element = -1


    position = []
    velocity = []
    personalBest = 100
    personalBestPosition = []
    informantIndex = -1
    group = -1
    element = -1


class NeuralNetwork:

    def __init__(self, *args):  # e.g.: NN(3,4,5,6,2)
        self.hidden_layer = list()
        self.weights = list()
        self.new_weights_attempt = list()
        # GLOBAL VARIABLE
        self.length_weight_matrix=0
        self.currentWeightList = []
        for n in range(len(args)):
            if n == 0:
                self.input_layer = Layer(args[n], "identity")
            elif n == len(args) - 1:
                self.output_layer = Layer(args[n], "sigmoid")
            else:
                self.hidden_layer.append(Layer(args[n], "sigmoid"))

        #make list of numbers
        num_nodes_hidden_layers = []
        for hidden_layer in range(0, self.hidden_layer.__len__()):
            num_nodes_hidden_layers.append(self.hidden_layer[hidden_layer].number_of_nodes)

        #fill list of numbers with values of hidden layers
        self.weight_matrix_num_nodes = 0
        self.weight_matrix_num_nodes = self.input_layer.number_of_nodes*num_nodes_hidden_layers[0]

        if num_nodes_hidden_layers.__len__() > 1:
            for i in range(0, num_nodes_hidden_layers.__len__() - 1):
                self.weight_matrix_num_nodes = self.weight_matrix_num_nodes + num_nodes_hidden_layers[i]*num_nodes_hidden_layers[i + 1]

        self.weight_matrix_num_nodes = self.weight_matrix_num_nodes + self.output_layer.number_of_nodes*num_nodes_hidden_layers[-1]


    def assign_weights(self):
        prev_layer = self.input_layer
        for layer in self.hidden_layer:
            self.weights.append(np.random.rand(layer.number_of_nodes, prev_layer.number_of_nodes))
            prev_layer = layer
        self.weights.append(np.random.rand(self.output_layer.number_of_nodes, prev_layer.number_of_nodes))
        print(self.weights)


    def assign_weights_from_pso(self, position):
        # are the rows & columns being reshpaed in the right way ?? ??? ?
        self.new_weights_attempt = []
        self.new_weights_attempt.append(np.array([position[0], position[1], position[2]]).reshape(3,1))
        self.new_weights_attempt.append(np.array([position[3], position[4], position[5]]).reshape(1,3))
        self.weights = self.new_weights_attempt



        #print(self.weights)

        return self.weights


    def feed_forward(self):
        inputs = np.array([0.8])
        actual_output = np.array([.71])

        #print(self.weights)

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
        #print(mse)

        return mse[0]

    # So i think the position matrix is just the weight matrix. Then, each particle (insect) in the swarm
    # has its own neural network, so its own weights matrix, position, velocity matrix etc.

    def PSO(self, numOfParticles):

        insects = 15
        alpha = 0.75
        beta = 2
        gamma = 2
        delta = 0.1
        stepsize = 1

        # Global fest fitness value found

        best = 100
        bestPosition = []

        particles = [Particle() for n in range(numOfParticles)]


        # Make a flat numpy array of the weight matrices in an array called positions

        velocities = []
        #for i in range(0, weight_flattened.__len__()):
            #velocities.append(round(random.uniform(-1, 1), 3))

        # change pos and vel for each particle to random
        for particle in range(0,particles.__len__()):
            for position in range(0, self.weight_matrix_num_nodes ):
                particles[particle].position.append(round(random.uniform(-1, 1), 3))
                particles[particle].velocity.append(round(random.uniform(-1, 1), 3))
        #print(particles[0].position)
        #print(particles.__len__())



        for particle in range(particles.__len__()):
            #set weights by sending positions to pso
            self.assign_weights_from_pso(particles[particle].position)
            self.currentWeightList = [particles[particle].position]

            # Get the mean square error for each of these particle
            currentMSE = self.feed_forward()
            # currentPosition = self.weights
            # print(a)
            #print(self.weights)

            #record personal best of particle
            if(currentMSE < particles[particle].personalBest):
                particles[particle].personalBest = currentMSE
                particles[particle].personalBestPosition = self.currentWeightList


            #compare mse with current best
            if( currentMSE< best):
                best = currentMSE
                bestPosition = self.currentWeightList

            #print(best)

        #This is making the informants. Our plan was split the insects into 4 groups, and the first ~quarter in the
        #first group, 2nd etc etc. Then each group looks at the member of the group with the lowest MSE achieved, points
        #to him and makes his personal best the informants best

        #copy array
        tempParticles = particles

        firstQuarter= int(particles.__len__() * 0.25)
        secondQuarter = firstQuarter *2
        thirdQuarter = particles.__len__() - firstQuarter

        group1 = []
        group2 = []
        group3 = []
        group4 = []


        group1 = tempParticles[0:firstQuarter]
        group2 = tempParticles[firstQuarter:secondQuarter]
        group3 = tempParticles[secondQuarter:thirdQuarter]
        group4 = tempParticles[thirdQuarter:particles.__len__()]

        groupList = [group1, group2, group3, group4]

       # print(group1[0].personalBest, group1[1].personalBest, group1[2].personalBest)

        #save temporary group informant index (index of the informant in the group)
        groupInformantIndex = -1

        #Loop through each group, reset the MSE, Calculate MSE and the particle with the best MSE is the informant (and has the informantindex)
        group1MSE = 10
        for g in range(group1.__len__()):
            if group1[g].personalBest < group1MSE:
                groupInformantIndex = g
        for g in range(group1.__len__()):
            group1[g].informantIndex = groupInformantIndex
            group1[g].group = 0
            group1[g].element = g

        group1MSE = 10
        for g in range(group2.__len__()):
            if group2[g].personalBest < group1MSE:
                groupInformantIndex = g
        for g in range(group2.__len__()):
            group2[g].informantIndex = groupInformantIndex
            group2[g].group = 1
            group2[g].element = g


        group1MSE = 10
        for g in range(group3.__len__()):
            if group3[g].personalBest < group1MSE:
                groupInformantIndex = g
        for g in range(group3.__len__()):
            group3[g].informantIndex = groupInformantIndex
            group3[g].group = 2
            group3[g].element = g


        group1MSE = 10
        for g in range(group4.__len__()):
            if group4[g].personalBest < group1MSE:
                groupInformantIndex = g
        for g in range(group4.__len__()):
            group4[g].informantIndex = groupInformantIndex
            group4[g].group = 3
            group4[g].element = g


        b =0
        c = 0
        d = 0

        epsilon = .1

        #calculating v_i's,plan here is there's like 15 insects & each one has an array of all its weight matrices
        #lets say 6 [w_0, ..., w_6], we want to calculate the new v_i which is each w_n for each insect, doing all
        #the adjustments based on the appropriate changes

        for particle in range(particles.__len__()):
            print(particles[particle].position)
            for vel in range(particles[particle].velocity.__len__()):
                b = round(random.uniform(0, beta),3)
                c = round(random.uniform(0, gamma),3)
                d = round(random.uniform(0, delta),3)
                i = groupList[particles[particle].group][particles[particle].element].informantIndex
                inf = groupList[particles[particle].group][i].personalBestPosition[0][vel]
                #print(inf)

                #print(particles[particle].velocity[vel])
                #print(particles[particle].personalBestPosition[0][vel])
                particles[particle].velocity[vel] = round((alpha * particles[particle].velocity[vel]) +\
                                                    (b * (particles[particle].personalBestPosition[0][vel] - (particles[particle].position[vel]))) +\
                                                    (c * (inf - (particles[particle].position[vel]))) +\
                                                    (d * ((bestPosition[0][vel]) - (particles[particle].position[vel]))),3)

                #print(particles[particle].velocity[vel])
            for pos in range(particles[particle].position.__len__()):
                particles[particle].position[pos] = round(particles[particle].position[pos] + epsilon * particles[particle].velocity[pos],3)
            print(particles[particle].velocity)
            print(particles[particle].position)
            print('-------------------------------')

    def __repr__(self):
        return "Input Layer: " + str(self.input_layer) + "\n" + "Hidden layers: " + \
               str(self.hidden_layer) + "\n" + "Output Layers: " + str(self.output_layer)


nn = NeuralNetwork(1, 3, 1)

#nn.assign_weights()

#nn.feed_forward()

nn.PSO(15)

#nn.assign_weights_from_pso()

