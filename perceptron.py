from numpy import exp, array, random, dot, tanh
import csv
import os
import shutil

def data_to_scv(path_with_name, data):
    with open(f'{path_with_name}', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'w', 'y', 'sum_errors']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            writer.writerow(d)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


class Perceptron():
    def __init__(self, mod=None):
        self.mod = mod

        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((4, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __tanh(self, x):
        # t = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return (1/2)*(tanh(x) + 1)

    def __binary_step(self, x):
        scale = lambda x: 0 if x < 0 else 1
        return array([array([scale(x)]) for x in x])

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, sigmoid_x):
        return sigmoid_x * (1 - sigmoid_x)

    def __tanh_derivative(self, tanh_x):
        return 1 - tanh_x ** 2

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, nu=0.3):
        if self.mod is None:
            self.mod = 1

        path = "./analyse"
        createFolder(path)
        # fieldnames = ['epoch', 'w', 'y', 'sum_errors']
        data = []

        print("Training:\n")

        output = None
        error = None
        adjustment = None

        for iteration in range(number_of_training_iterations):
            d = {}
            d['epoch'] = iteration

            if self.mod == 1:   # BINARY_STEP
                # Pass the training set through our neural network (a single neuron).
                output = self.think(training_set_inputs)

                # Calculate the error (The difference between the desired output
                # and the predicted output).
                error = training_set_outputs - output

                # Multiply the error by the input and again by the gradient of the Sigmoid curve.
                # This means less confident weights are adjusted more.
                # This means inputs, which are zero, do not cause changes to the weights.
                # adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
                # adjustment = dot(training_set_inputs.T, (1 - output) * error * output)
                adjustment = dot(training_set_inputs.T, nu * error)

                # Adjust the weights.
                self.synaptic_weights += adjustment

            elif self.mod == 2:     # TANH

                output = self.think(training_set_inputs)
                error = training_set_outputs - output
                adjustment = dot(training_set_inputs.T, nu * error * self.__tanh_derivative(output))
                self.synaptic_weights += adjustment

            elif self.mod == 3:     # Sigmoid

                output = self.think(training_set_inputs)
                error = training_set_outputs - output
                adjustment = dot(training_set_inputs.T, nu * error * self.__sigmoid_derivative(output))
                self.synaptic_weights += adjustment

            d['w'] = [x[0] for x in self.synaptic_weights]
            d['y'] = [x[0] for x in output]
            d['sum_errors'] = sum(error.T[0])
            data.append(d)
            print('>epoch=%d, lrate=%.3f, error=%.3f\n' % (iteration, nu, sum(error.T[0])))


        path_with_name = f"{path}/{self.mod}.csv"
        data_to_scv(path_with_name, data)

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        if self.mod == 1:
            return self.__binary_step(dot(inputs, self.synaptic_weights))

        elif self.mod == 2:
            return self.__tanh(dot(inputs, self.synaptic_weights))

        elif self.mod == 3:
            return self.__sigmoid(dot(inputs, self.synaptic_weights))


def simple_test(mod=None, arr=None, epoch=None, nu=None):
    if mod is None:
        mod = 1

    if arr is None:
        arr = [0, 0, 0, 1]

    if epoch is None or nu is None:
        epoch = 4
        nu = 0.3

    neural_network = Perceptron(mod)
    print(f"Random starting synaptic weights: {neural_network.synaptic_weights}\n")

    training_set_inputs = array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ])

    training_set_outputs = array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]]).T
    neural_network.train(training_set_inputs, training_set_outputs, epoch, nu)
    print(f"New synaptic weights after training: {neural_network.synaptic_weights}\n")
    print(f"Considering new situation {arr} -> {neural_network.think(array(arr))}\n")

def big_test(mod=None, epoch=None, nu=None):
    if mod is None:
        mod = 1

    if epoch is None or nu is None:
        epoch = 4
        nu = 0.3

    neural_network = Perceptron(mod)
    training_set_inputs = array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ])

    training_set_outputs = array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]]).T
    neural_network.train(training_set_inputs, training_set_outputs, epoch, nu)

    # Test
    test_set_inputs = array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])

    tset_set_outputs = array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    accurency = 0
    errors = 0

    if mod == 1:
        for i in range(len(test_set_inputs)):
            if neural_network.think(test_set_inputs[i]) == tset_set_outputs[i]:
                accurency += 1
            else:
                errors += 1

    elif mod == 2 or mod == 3:
        for i in range(len(test_set_inputs)):
            # print( int( list( neural_network.think(test_set_inputs[i]) )[0] ) )
            ans = None
            if neural_network.think(test_set_inputs[i]) > array([0.7]):
                ans = array([1])
            else:
                ans = array([0])

            if ans == tset_set_outputs[i]:
                accurency += 1
            else:
                errors += 1

    print(f"Accurency: {accurency * 100 / 16}%\nErrors: {errors * 100 / 16}%\n")


if __name__ == "__main__":
    # simple_test(mod=1, epoch=4, nu=0.3)
    big_test(mod=2, epoch=10, nu=0.1)
    #BINARY_STEP: big_test(mod=1, epoch=6, nu=0.2) # -> 87.5%/12.5%
    #TANH: big_test(mod=2, epoch=10, nu=0.1) # -> 93.75%/6.25%
    #Sigmoid: big_test(mod=3, epoch=51, nu=0.1) # -> 93.75%/6.25%

"""
    training_set_inputs = array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])

    training_set_outputs = array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]]).T
    
    LINKS:
    https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
    https://pythonmachinelearning.pro/perceptrons-the-first-neural-networks/
    https://towardsdatascience.com/perceptron-and-its-implementation-in-python-f87d6c7aa428
    https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3
    https://ru.wikipedia.org/wiki/%D0%94%D0%B5%D0%BB%D1%8C%D1%82%D0%B0-%D0%BF%D1%80%D0%B0%D0%B2%D0%B8%D0%BB%D0%BE
    https://neurohive.io/ru/tutorial/prostaja-nejronnaja-set-python/
    https://python-scripts.com/intro-to-neural-networks
    https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
    https://towardsdatascience.com/implementing-different-activation-functions-and-weight-initialization-methods-using-python-c78643b9f20f
"""
