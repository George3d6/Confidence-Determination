import random
import pickle
import os

import numpy as np
from ordered_set import OrderedSet


random.seed(1)

def gen_a(mode, set, degree):
    pct = 3
    length = min(int(pow(10,degree)/pct),pow(10,4))
    try:
        X = pickle.load(open(f'cache/a_X_{mode}_{degree}_{length}_{set}.pickle', 'rb'))
        Yn = pickle.load(open(f'cache/a_Yn_{mode}_{degree}_{length}_{set}.pickle', 'rb'))
        return X, Yn
    except:
        pass

    exists = []
    X = [[random.randint(1,10) for _ in range(degree)] for _ in range(length)]

    for i in range(len(X)):
        n = 0
        while OrderedSet(X[i]) in exists:
            X[i] = [random.randint(1,10) for _ in range(degree)]

        exists.append(OrderedSet(X[i]))

    Y = []
    for x in X:
        y = 0
        for i in range(len(x)):
            if mode == 'polynomial':
                y += pow(x[i],(i+1))
            elif mode == 'linear':
                y += x[i] * (i+1)
            elif mode == 'polynomial_w_coef':
                y += pow(x[i],(i+1)) * (degree - i)
        Y.append(y)

    lim1 = np.quantile(Y,1/3)
    lim2 = np.quantile(Y,2/3)

    Yn = []
    for y in Y:
        if y < lim1:
            Yn.append(0)
        elif y < lim2:
            Yn.append(1)
        else:
            Yn.append(2)

    pickle.dump(X, open(f'cache/a_X_{mode}_{degree}_{length}_{set}.pickle', 'wb'))
    pickle.dump(Yn, open(f'cache/a_Yn_{mode}_{degree}_{length}_{set}.pickle', 'wb'))
    return X,Yn

def gen_b(mode, set, degree):
    pct = 3
    length = min(int(pow(10,degree)/pct),pow(10,4))
    try:
        X = pickle.load(open(f'cache/b_X_{mode}_{degree}_{length}_{set}.pickle', 'rb'))
        Yn = pickle.load(open(f'cache/b_Yn_{mode}_{degree}_{length}_{set}.pickle', 'rb'))
        return X, Yn
    except:
        pass

    exists = []
    X = [[random.randint(1,10) for _ in range(degree)] for _ in range(length)]

    for i in range(len(X)):
        n = 0
        while OrderedSet(X[i]) in exists:
            X[i] = [random.randint(1,10) for _ in range(degree)]

        exists.append(OrderedSet(X[i]))

    Y = []
    for x in X:
        y = 0
        for i in range(len(x)):
            if mode == 'polynomial':
                y += pow(x[i],(i+1))
            elif mode == 'linear':
                y += x[i] * (i+1)
            elif mode == 'polynomial_w_coef':
                y += pow(x[i],(i+1)) * (degree - i)
        Y.append(y)

    lim1 = np.quantile(Y,1/2)

    Yn = []
    for i, y in enumerate(Y):
        if X[i][0] == 5:
            Yn.append(2)
        elif y < lim1:
            Yn.append(0)
        else:
            Yn.append(1)

    pickle.dump(X, open(f'cache/b_X_{mode}_{degree}_{length}_{set}.pickle', 'wb'))
    pickle.dump(Yn, open(f'cache/b_Yn_{mode}_{degree}_{length}_{set}.pickle', 'wb'))
    return X,Yn

def gen_c(mode, set, degree):
    pct = 3
    length = min(int(pow(10,degree)/pct),pow(10,4))
    X, Yn = gen_a(mode, set,degree)
    for i in range(len(Yn)):
        if Yn[i] == 2:
            if random.randint(1,3) == 3:
                Yn[i] = 0
    return X, Yn

if __name__ == '__main__':
    for f in (gen_a, gen_b, gen_c):
        X,Y = f('polynomial_w_coef','train',4)
        y_dict = {}
        for y in Y:
            if y not in y_dict:
                y_dict[y] = 0
            y_dict[y] += 1
        for y in y_dict:
            y_dict[y] = str(round( (100*y_dict[y]/len(Y)) ,2)) + '%'

        print(f'Y distribution for function {f}')
        print(y_dict)
