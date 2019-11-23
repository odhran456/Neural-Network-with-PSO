import numpy as np
import random
import matplotlib.pyplot as plt


class AFs:
    # Returns value after calculating the value through the passed activation function

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def step(x):
        return 0 if x < 0 else 1

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def gaussian(x):
        return np.exp(- ((x**2)/(2)))


class Neuron:

    def __init__(self, af_name):
        # Sets activation function of Neuron
        self.af_name = af_name

        if af_name == "identity":
            self.activation_fn = AFs.identity
        elif af_name == "step":
            self.activation_fn = AFs.step
        elif af_name == "sigmoid":
            self.activation_fn = AFs.sigmoid
        elif af_name == "tanh":
            self.activation_fn = AFs.tanh
        elif af_name == "gaussian":
            self.activation_fn = AFs.gaussian
        elif af_name == "cos":
            self.activation_fn = AFs.cos

    # Returns activation function of Neuron
    def get_af(self, dot_prod):
        return self.activation_fn(dot_prod)

    # Changes activation function of Neuron
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
        elif new_af_name == "tanh":
            self.activation_fn = AFs.tanh
            self.af_name = "tanh"
        elif new_af_name == "cos":
            self.activation_fn = AFs.cos
            self.af_name = "cos"


    def __repr__(self):
        return self.af_name


class Layer:

    def __init__(self, num_nodes, afname):
        self.number_of_nodes = num_nodes
        self.neuron_list = [Neuron(afname) for neuron in range(num_nodes)]
    # Returns activation function of layer
    def activate_neuron(self, index, dp):
        return self.neuron_list[index].get_af(dp)
    # Changes activation function of layer
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
        self.element = -1       # Index of the group
        self.currentMSE = 0


    position = []
    velocity = []
    personalBest = 100
    personalBestPosition = []
    informantIndex = -1
    group = -1
    element = -1
    currentMSE = 0


class NeuralNetwork:

    def __init__(self, *args):
        # Initialize HNN variables
        self.hidden_layer = list()
        self.weights = list()
        self.new_weights_attempt = list()
        self.length_weight_matrix=0
        self.currentWeightList = []

        # Set the first and last args given as the input and output layers
        # Every element between the first and last arg is a hidden layer
        for n in range(len(args)):
            if n == 0:
                self.input_layer = Layer(args[n], "identity")
            elif n == len(args) - 1:
                self.output_layer = Layer(args[n], "tanh")
            else:
                self.hidden_layer.append(Layer(args[n], "sigmoid"))

        # Make list of values for Hidden layer
        num_nodes_hidden_layers = []
        for hidden_layer in range(0, self.hidden_layer.__len__()):
            num_nodes_hidden_layers.append(self.hidden_layer[hidden_layer].number_of_nodes)

        # Fill list of numbers with values of hidden layers
        self.weight_matrix_num_nodes = 0
        self.weight_matrix_num_nodes = self.input_layer.number_of_nodes*num_nodes_hidden_layers[0]
        if num_nodes_hidden_layers.__len__() > 1:
            for i in range(0, num_nodes_hidden_layers.__len__() - 1):
                self.weight_matrix_num_nodes = self.weight_matrix_num_nodes + num_nodes_hidden_layers[i]*num_nodes_hidden_layers[i + 1]

        self.weight_matrix_num_nodes = self.weight_matrix_num_nodes + self.output_layer.number_of_nodes*num_nodes_hidden_layers[-1]

    # Assigns positions it recieves as a parameter to the weights of the PSO and Returns the Weights
    def assign_weights_from_pso(self, position):
        # TODO: fix the hardcoding of indeces of positions to the weights of the PSO
        self.new_weights_attempt = []
        self.new_weights_attempt.append(np.array([position[0], position[1], position[2]]).reshape(3,1))
        self.new_weights_attempt.append(np.array([position[3], position[4], position[5]]).reshape(1,3))
        self.weights = self.new_weights_attempt

        return self.weights

    # Recieves input and output values and returns the MSE and estimated output of the HNN
    def feed_forward(self,input,output):
        # Recieve the current input and output from the file, passed as parameters
        inputs = np.array([float(input)])
        actual_output = np.array([float(output)])

        # This makes the column vector of node values for the current layer. It takes the initial layer (inputs)
        # and does the dot product with the first array of weights and that gives the next layer on node values
        currentMatrix = self.hidden_layer[0].activate_neuron(0, np.dot(self.weights[0], inputs))

        # It's put into a numpy array as in the list it had a null column (shape (n, ) )
        # so it is reshaped into a column vector
        tempMatrix = np.array([currentMatrix]).transpose()

        # Sets a previous Matrix
        prev_matrix = tempMatrix

        # If there are more than 1 hidden layers, loop through them and perform the matrix multiplications
        if self.hidden_layer.__len__() > 1:
            for i in range(1, self.weights.__len__() - 1):
                tempMatrix = self.hidden_layer[i].activate_neuron(0, np.dot(self.weights[i], prev_matrix))
                prev_matrix = tempMatrix

        # This holds the estimated output of HNN
        estimated_output = self.output_layer.activate_neuron(0, np.dot(self.weights[-1], tempMatrix))

        # Calculates the MSE between the estimated output of the HNN and the actual output from the file
        sum = 0
        for i in range(0, estimated_output.shape[0]):
            sum = sum + ((actual_output[i] - estimated_output[i])**2)

        mse = np.sqrt(sum / estimated_output.shape[0])

        # returns the MSE and the estimated output.
        return mse[0], estimated_output


    def PSO(self):

        # The 6 Hyperparameters of the PSO are defined here
        numOfParticles = 55
        alpha = 0.75
        beta = 4
        gamma = 3
        delta = 0.5
        stepSize = 2

        # stepSize loop variables
        currentStep = 0         # The current step the stepSize loop is in.
        bestTarget = 0.005      # The stepSize loop will stop prematurely if there is a MSE found that was better than this value.

        # Debugging variables
        alltimeBest = 100       # The all time best MSE of all the particles.

        # List that hold the input and output data of a given file.
        inputs = self.Read_Data(True)
        outputs = self.Read_Data(False)

        # Plot variables
        plotList = []           # List that will hold the values to plot, it will hold all the outputs of the HNN

        # START ITERATION THROUGH EACH INPUT AND OUTPUT

        # Loops through each input that is given in the inputs[] array, which reads data from a file.
        for index in range(inputs.__len__()):
            # List of particles with the
            particles = [Particle() for n in range(numOfParticles)]

            # START PSO INITIALIZATION

            # Randomizes the position and velocity of each particle in the particles list
            for particle in range(0,particles.__len__()):
                for position in range(0, self.weight_matrix_num_nodes ):
                    particles[particle].position.append(round(random.uniform(-2, 2), 3))
                    particles[particle].velocity.append(round(random.uniform(-2, 2), 3))

            # END PSO INITIALIZATION

            #Variables that store the best MSE and the corresponding output from the HNN
            best = 100
            bestEstimatedOutput = 100

            # START ITERATION THROUGH STEPSIZES

            while(best > bestTarget or currentStep < stepSize ):
                currentStep = currentStep + 1

                # loop through each particle's positions, assign the HNN weights to this position
                # Run the feedforward with the new weights and record the best feedforward value, MSE and position for each particle
                # and also record the global best feedforward value, MSE and position.
                for particle in range(particles.__len__()):
                    # Set weights by sending positions to pso
                    self.assign_weights_from_pso(particles[particle].position)
                    self.currentWeightList = [particles[particle].position]

                    # Get the MSE
                    currentMSE = self.feed_forward(inputs[index], outputs[index])
                    particles[particle].currentMSE = currentMSE[0]

                    # Check if personal best of particle is exceeded if so record it
                    if(currentMSE[0] < particles[particle].personalBest):
                        particles[particle].personalBest = currentMSE[0]
                        particles[particle].personalBestPosition = self.currentWeightList


                    #Compare MSE with current global best MSE
                    if( currentMSE[0]< best):
                        best = currentMSE[0]
                        bestEstimatedOutput = currentMSE[1]
                        bestPosition = self.currentWeightList

                # START FORMING GROUPS AND ASSIGNING INFORMANTS

                # This is making the informants. Our plan was split the particle into 4 groups, and the first ~quarter in the
                # first group, 2nd etc etc. Then each group looks at the member of the group with the (best) lowest MSE achieved,
                # this particle becomes the informant and everyone in the group follows this informant.

                # Checking the particles array size and split it into 4 quarters
                firstQuarter= int(particles.__len__() * 0.25)
                secondQuarter = firstQuarter *2
                thirdQuarter = particles.__len__() - firstQuarter

                # Creating and filling the 4 groups the particles will be split into
                group1 = particles[0:firstQuarter]
                group2 = particles[firstQuarter:secondQuarter]
                group3 = particles[secondQuarter:thirdQuarter]
                group4 = particles[thirdQuarter:particles.__len__()]

                # Create a list that hold all the groups and their values
                groupList = [group1, group2, group3, group4]

                # Initialize variables used to loop through groups
                groupInformantIndex = -1    # Initialize the index in which the group informant lies within the group
                groupMSE = 10               # Initialize the index in which the group informant lies within the group

                # LOOP 1
                # Loop through each group, reset the groupMSE variable
                # Calculate MSE and the particle and choose the particle with the best MSE as the informant

                # LOOP 2
                # Assign everyone in the group the informant and pass them the index of their informant within the group

                # Loop through group 1
                for g in range(group1.__len__()):
                    if group1[g].personalBest < groupMSE:
                        groupInformantIndex = g
                        groupMSE = group1[g].personalBest
                for g in range(group1.__len__()):
                    group1[g].informantIndex = groupInformantIndex
                    group1[g].group = 0
                    group1[g].element = g

                # Loop through group 2
                groupMSE = 10
                for g in range(group2.__len__()):
                    if group2[g].personalBest < groupMSE:
                        groupInformantIndex = g
                        groupMSE = group2[g].personalBest
                for g in range(group2.__len__()):
                    group2[g].informantIndex = groupInformantIndex
                    group2[g].group = 1
                    group2[g].element = g

                # Loop through group 3
                groupMSE = 10
                for g in range(group3.__len__()):
                    if group3[g].personalBest < groupMSE:
                        groupInformantIndex = g
                        groupMSE = group3[g].personalBest
                for g in range(group3.__len__()):
                    group3[g].informantIndex = groupInformantIndex
                    group3[g].group = 2
                    group3[g].element = g

                # Loop through group 4
                groupMSE = 10
                for g in range(group4.__len__()):
                    if group4[g].personalBest < groupMSE:
                        groupInformantIndex = g
                        groupMSE = group4[g].personalBest
                for g in range(group4.__len__()):
                    group4[g].informantIndex = groupInformantIndex
                    group4[g].group = 3
                    group4[g].element = g

                # END FORMING GROUPS AND ASSIGNING INFORMANTS

                #Initialize and set Epsilon
                epsilon = .1

                # START CALCULATE NEW POSITIONS

                # For each particle change the velocity and update the position by calculating Vi
                for particle in range(particles.__len__()):
                    # Change the velocity of each particle
                    for vel in range(particles[particle].velocity.__len__()):
                        # Randomize b, c, d to a number between 0 and beta, gamma, delta respectively.
                        b = round(random.uniform(0, beta),3)
                        c = round(random.uniform(0, gamma),3)
                        d = round(random.uniform(0, delta),3)

                        informantIndex = groupList[particles[particle].group][particles[particle].element].informantIndex
                        informantPersonalBestPosition = groupList[particles[particle].group][informantIndex].personalBestPosition[0][vel]
                        particles[particle].velocity[vel] = round((alpha * particles[particle].velocity[vel]) +\
                                                            (b * (particles[particle].personalBestPosition[0][vel] - (particles[particle].position[vel]))) +\
                                                            (c * (informantPersonalBestPosition - (particles[particle].position[vel]))) +\
                                                            (d * ((bestPosition[0][vel]) - (particles[particle].position[vel]))),3)

                    # Calculate Vi for each position in a particle
                    for pos in range(particles[particle].position.__len__()):
                        particles[particle].position[pos] = round(particles[particle].position[pos] + epsilon * particles[particle].velocity[pos],3)

                # END CALCULATE NEW POSITIONS

                #Safety check in case currentStepSize surpases stepSize and creates an infinite loop
                if(currentStep > stepSize or currentStep == stepSize):
                    break

            # END ITERATION THROUGH STEPSIZES

            # Add bestEstimatedOutput for this INPUT into the plotList
            plotList.append(bestEstimatedOutput[0][0])

            # update and save allTimeBest MSE if better, for debugging purposes
            if(best < alltimeBest):
                alltimeBest = best

        # END ITERATION THROUGH EACH INPUT AND OUTPUT

        # Plot the HNN results against the actual Input and Output values
        plt.scatter(inputs, plotList, c='r')
        plt.scatter(inputs, outputs)
        plt.show()

# Reads the data from a textfile, puts the data into a list and returns this list.
# The Parameter returnTypeInput determines it you want to return the inputs or outputs as a list.
    def Read_Data(self, returnTypeInput):
        filename = 'C:/Users/loren/Documents/HeriotWatt/Bio/1in_cubic.txt'
        f = open(filename, "r").read().replace("   ", "\n")
        mylist = f.split("\n")

        input = []
        output = []

        dir = 1
        for i in range(mylist.__len__()):
            #if(i == 0):
            #    continue

            if mylist[i] != '':
                if dir > 0:
                    input.append(float(mylist[i]))
                else:
                    output.append(float(mylist[i]))
                dir = dir * -1

       # print (input)
       # print (output)

        if returnTypeInput == True:
            return input
        else:
            return output

# MAIN FUNCTION
nn = NeuralNetwork(1, 3, 1)
nn.PSO()
