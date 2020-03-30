from release.perceptron_lib import *

def write_min(combination, outputFile, W):
    file = open(outputFile, 'w')
    file.write('set:\n')
    for i,x in enumerate(combination[1]):
        file.write('X(' + str(i+1) + ') = (' + str(x[0])[4:-1] + ')')
        file.write('\n')
    file.write('\n')

    nu = 0.3
    E = 1
    k = -1

    arrayE = list()
    arrayK = list()

    Y = list()

    F = [i[1] for i in combination[1]]

    while E != 0:
        k += 1

        prev_W = list(W)

        for (x, f) in combination[1]:
            n = net(W, x)

            y = think(n)
            Y.append(y)

            d = delta(f, y)

        E = totalError(Y, F)

        write_Data(file, k, Y, prev_W, E)

        Y = list()

        if E != 0:
            for (x,f) in combination[1]:
                n = net(W, x)

                y = think(n)

                d = delta(f, y)

                W = recount_W(W, x, d, n, nu)

        arrayK.append(k)
        arrayE.append(E)

    drawGraph(arrayE, arrayK, outputFile)

def check_combination(W, combination):
    nu = 0.3
    E = 1
    k = -1

    Y = list()

    epochs = 200
    while E != 0 and k < epochs:
        k += 1

        e = 0

        prev_W = list(W)

        for (x, f) in combination:
            n = net(W, x)

            y = think(n)

            d = delta(f, y)

            if d != 0:
                e += 1

        if e != 0:
            e = 0
            for (x,f) in combination:
                n = net(W, x)

                y = think(n)

                d = delta(f, y)

                if d != 0:
                    e += 1

                W = recount_W(W, x, d, n, nu)
        E = e
    if k < epochs:
        return W, k
    else:
        return W, -1

def find_min_vector_tanh(inputW, inputF, outputFile):
    best_combination = None
    X = bin_generation(4)

    for i in range(2**4, 2, -1):
        combinations = list(itertools.combinations(zip(X, inputF), i))

        arrayKN = list()

        for combination in combinations:

            Y = list()

            W, k = check_combination(list(inputW), combination)

            if k != -1:

                for (x, f) in zip(X, inputF):
                    n = net(W, x)
                    y = think(n)
                    Y.append(y)

                E = totalError(Y, inputF)

                if E == 0:
                    arrayKN.append((k, combination, W))
                    best_combination = sorted(arrayKN, key = lambda education: education[0])[0]

    write_min(best_combination, outputFile, list(inputW))

def train(W, F, outputFile):
    X = bin_generation(4)
    nu = 0.3
    E = 1
    k = -1

    arrayE = list()
    arrayK = list()
    Y = list()

    file = open(outputFile, 'w')

    while E != 0:
        k += 1

        prev_W = list(W)

        for (x, f) in zip(X, F):
            n = net(W, x)

            y = think(n)
            Y.append(y)

            d = delta(f, y)

        E = totalError(Y, F)

        write_Data(file, k, Y, prev_W, E)

        Y = list()

        if E != 0:

            for (x,f) in zip(X, F):

                n = net(W, x)

                y = think(n)

                d = delta(f, y)

                W = recount_W(W, x, d, n, nu)

        arrayK.append(k)
        arrayE.append(E)
        # drawGraph(arrayE, arrayK, outputFile)

def train_tanh():
    outputFile = "tanh"

    W = [0, 0, 0, 0, 0]
    F = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    train(W, F, outputFile + '_logistics')

def find_min_vector():
    outputFile = "find_min_vector_tanh"

    W = [0, 0, 0, 0, 0]
    F = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    find_min_vector_tanh(W, F, outputFile + '_logistics')

if __name__ == '__main__':
    train_tanh()
    # find_min_vector()