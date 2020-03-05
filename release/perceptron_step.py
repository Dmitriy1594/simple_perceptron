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

    def round(self, x):
        if x >= 0.5:
            return 1
        elif x < 0.5:
            return 0

    def __tanh(self, net):
        return (1/2)*(tanh(net) + 1)

    def __tanh_derivative(self, tanh_x):
        return  1 - (1 / (2 * (cos(tanh_x)**2)))

    def __hamming(self, input_y, output_aim):
        # return sum([x for x in list(map(lambda x: x[0]+x[1], list(zip(input_y, output_aim)))) if x == 1])
        E = 0
        for (y, f) in zip(input_y, output_aim):
            if y != f:
                E += 1
        return E

    def __delta_weight(self, error, xi):
        return self.nu * error * xi

    def train(self, training_set_inputs, training_set_outputs):
        training_set_input = training_set_inputs.copy()

        print("Training:\n")

        # k - эпоха обучения
        k = int()
        y = int()
        error = int()
        E = 1
        output_y = []

        while E != 0:

            output_y = []

            # 1 получаем реальный выход нейрона
            for l in range(len(training_set_input)):

                y = self.think(training_set_input[l])

                output_y.append(y)

            E = self.__hamming(output_y, training_set_outputs)

            if self.data_to_scv is True:
                d = {}
                d['k'] = k
                d['w'] = self.synaptic_weights.copy()
                d['y'] = output_y
                d['E'] = E
                self.data.append(d)
                print(f'>epoch={k}, weights={self.synaptic_weights}, E={E}\n')

            # 2 обновление весов
            # l - шаг обучения
            for l in range(len(training_set_input)):

                y = self.think(training_set_input[l])

                error = training_set_outputs[l] - y

                x14 = training_set_input[l]
                x04 = [1] + x14

                for it in range(len(x04)):
                    delta = self.__delta_weight(error, x04[it])
                    self.synaptic_weights[it] += delta

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

    def think(self, inputs):
        net = sum(list(map(lambda x: x[0]*x[1], list(zip(inputs, self.synaptic_weights[1:])))))
        net += self.synaptic_weights[0]
        return self.__binary_step(net)

    def __check_combination(self, combination):
        E = 1
        k = -1

        epochs = 200
        while E != 0 and k < epochs:
            k += 1

            e = 0

            for (x, f) in combination:
                y = self.think(x)

                d = f - y

                if d != 0:
                    e += 1

            if e != 0:
                e = 0
                for (x, f) in combination:
                    y = self.think(x)

                    d = f - y

                    if d != 0:
                        e += 1

                    x14 = x
                    x04 = [1] + x14

                    for it in range(len(x04)):
                        delta = self.__delta_weight(d, x04[it])
                        self.synaptic_weights[it] += delta
            E = e
        if k < epochs:
            return k
        else:
            return -1

    def find_min_vector(self, training_set_inputs, training_set_outputs):
        if self.synaptic_weights != [0, 0, 0, 0, 0]:
            self.synaptic_weights = [0, 0, 0, 0, 0]

        best_combination = None

        for i in range(16, 2, -1):
            self.synaptic_weights = [0, 0, 0, 0, 0]

            combinations = list(itertools.combinations(zip(training_set_inputs, training_set_outputs), i))

            arrayKN = list()

            for combination in combinations:

                self.synaptic_weights = [0, 0, 0, 0, 0]

                output_y = list()

                k = self.__check_combination(combination)

                if k != -1:

                    for l in range(len(training_set_inputs)):
                        y = self.think(training_set_inputs[l])

                        output_y.append(y)

                    E = self.__hamming(output_y, training_set_outputs)

                    if E == 0:
                        arrayKN.append((k, combination, self.synaptic_weights))
                        best_combination = sorted(arrayKN, key=lambda education: education[0])[0]

        print(best_combination)


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

def find_min_vector_step():
    neural_network = Perceptron("./find_min_vector_step")

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

    neural_network.find_min_vector(training_set_inputs, training_set_outputs)


if __name__ == "__main__":
    run_app()
    # find_min_vector_step()

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
