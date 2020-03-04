from release.perceptron_lib import *


class Perceptron():
    def __init__(self, path_to_save_file, data_to_scv=True):
        self.data_to_scv = data_to_scv

        self.synaptic_weights = [0, 0, 0, 0, 0]
        self.nu = 0.3

        if self.data_to_scv is True:
            self.data = []
            self.path = path_to_save_file
            createFolder(self.path)

    def __binary_step(self, x):
        scale = lambda x: 0 if x < 0 else 1
        return scale(x)

    def hamming(self, input_y, output_aim):
        return sum([x for x in list(map(lambda x: x[0]+x[1], list(zip(input_y, output_aim)))) if x == 1])

    def delta_weight(self, error, xi):
        return self.nu * error * xi

    def train(self, training_set_inputs, training_set_outputs):
        training_set_input = training_set_inputs.copy()

        print("Training:\n")

        # k - эпоха обучения
        k = int()
        y = int()
        error = int()
        ek = 1

        while ek != 0:

            output_y = []

            # l - шаг обучения
            for l in range(len(training_set_input)):

                y = self.think(training_set_input[l])

                output_y.append(y)

                error = training_set_outputs[l] - y

                x14 = training_set_input[l]
                x04 = [1] + x14

                for it in range(len(x04)):
                    delta = self.delta_weight(error, x04[it])
                    self.synaptic_weights[it] += delta

            ek = self.hamming(output_y, training_set_outputs)

            if self.data_to_scv is True:
                d = {}
                d['k'] = k
                d['w'] = self.synaptic_weights.copy()
                d['y'] = output_y
                d['E'] = ek
                self.data.append(d)
                print(f'>epoch={k}, weights={self.synaptic_weights}, E={ek}\n')

            k += 1

    # def train_random(self, training_set_inputs, training_set_outputs):
    #     training_set_input = training_set_inputs.copy()
    #     vectors = list(zip(training_set_input, training_set_outputs))
    #
    #     print("Training:\n")
    #
    #     # k - эпоха обучения
    #     k = int()
    #     y = int()
    #     error = int()
    #     ek = None
    #
    #     while ek != 0:
    #
    #         output_y = []
    #
    #         # l - шаг обучения
    #         for l in range(len(vectors)):
    #
    #             y = self.think(vectors[l][0])
    #
    #             output_y.append(y)
    #
    #             error = vectors[l][1] - y
    #
    #             x14 = vectors[l][0]
    #             x04 = [1] + x14
    #
    #             for it in range(len(x04)):
    #                 delta = self.delta_weight(error, x04[it])
    #                 self.synaptic_weights[it] += delta
    #
    #         ek = self.hamming(output_y, training_set_outputs)
    #
    #         if self.data_to_scv is True:
    #             d = {}
    #             d['k'] = k
    #             d['w'] = self.synaptic_weights.copy()
    #             d['y'] = output_y
    #             d['E'] = ek
    #             self.data.append(d)
    #             print(f'>epoch={k}, weights={self.synaptic_weights}, E={ek}\n')
    #
    #         k += 1
    #         random.seed(1)
    #         vectors = random.sample(vectors, len(vectors))

    def load_data_to_csv(self):
        if self.data_to_scv is True:
            path_with_name = f"{self.path}/step.csv"
            data_to_scv(path_with_name, self.data)
        else:
            print("WARNING: data_to_scv is False\n")

    # The neural network thinks.
    def think(self, inputs):
        net = sum(list(map(lambda x: x[0]*x[1], list(zip(inputs, self.synaptic_weights[1:])))))
        net += self.synaptic_weights[0]
        return self.__binary_step(net)


def find_min_vector():
    training_set_inputs = [
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
    ]

    training_set_outputs = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    # Создание зависимостей
    vectors = list(zip(training_set_inputs, training_set_outputs))

    for i in range(5):
        vectors.pop()

    # Перетасовка векторов
    vectors = random.sample(vectors, len(vectors))

    training_set_inputs = [x[0] for x in vectors]
    training_set_outputs = [x[1] for x in vectors]

    neural_network = Perceptron("./find_min_vector_step")
    neural_network.train(training_set_inputs, training_set_outputs)
    neural_network.load_data_to_csv()
    print(f"New synaptic weights after training: {neural_network.synaptic_weights}\n")

    # for it in range(1,16):
    #
    #
    #     if it == 5:
    #         break
    #
    #     # random.seed(1)
    #     # arr_inputs = random.sample(training_set_inputs, len(training_set_inputs))
    #     # arr_inputs = arr_inputs[it:]
    #     print()
    #
    #     neural_network = Perceptron("./find_min_vector_step")
    #     print(f"Random starting synaptic weights: {neural_network.synaptic_weights}\n")
    #
    #     training_set_outputs = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    #     # training_set_outputs = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    #     neural_network.train(training_set_inputs[it:], training_set_outputs)
    #     # neural_network.train_random(training_set_inputs, training_set_outputs)
    #     neural_network.load_data_to_csv()
    #     print(f"New synaptic weights after training: {neural_network.synaptic_weights}\n")

    # Test

    test_set_inputs = [
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
    ]

    tset_set_outputs = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    # tset_set_outputs = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    accurency = 0
    errors = 0

    for i in range(len(test_set_inputs)):
        if neural_network.think(test_set_inputs[i]) == tset_set_outputs[i]:
            accurency += 1
        else:
            errors += 1

    print(f"Accurency: {accurency * 100 / 16}%\nErrors: {errors * 100 / 16}%\n")


def run_app(arr=None):
    neural_network = Perceptron("./analyse_step")
    print(f"Random starting synaptic weights: {neural_network.synaptic_weights}\n")

    training_set_inputs = [
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
    ]

    if arr is None:
        arr = random.choice(training_set_inputs)

    training_set_outputs = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    # training_set_outputs = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    neural_network.train(training_set_inputs, training_set_outputs)
    # neural_network.train_random(training_set_inputs, training_set_outputs)
    neural_network.load_data_to_csv()
    print(f"New synaptic weights after training: {neural_network.synaptic_weights}\n")
    print(f"Considering new situation {arr} -> {neural_network.think(arr)}\n")

    # Test
    test_set_inputs = [
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
    ]

    test_set_outputs = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    # tset_set_outputs = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    accurency = 0
    errors = 0

    for i in range(len(test_set_inputs)):
        if neural_network.think(test_set_inputs[i]) == test_set_outputs[i]:
            accurency += 1
        else:
            errors += 1

    print(f"Accurency: {accurency * 100 / 16}%\nErrors: {errors * 100 / 16}%\n")




if __name__ == "__main__":
    run_app()
    # find_min_vector()

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
    
        training_set_inputs = [
        # [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        # [0, 0, 1, 1],
        [0, 1, 0, 0],
        # [0, 1, 0, 1],
        # [0, 1, 1, 0],
        # [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        # [1, 0, 1, 0],
        # [1, 0, 1, 1],
        # [1, 1, 0, 0],
        # [1, 1, 0, 1],
        # [1, 1, 1, 0],
        # [1, 1, 1, 1],
    ]
"""
