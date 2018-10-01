import numpy as np
import random

np.random.seed(42)
random.seed(42)

def import_data(file_path):
    data_target = []
    data_input = []
    with open(file_path, 'r') as f:
        while True:
            data_str = f.readline().split(' ')
            if not data_str[0]:
                break
            data_target.append(float(data_str[0]))
            data_input.append([float(a.split(':')[1]) for a in data_str[2:48]])

    x = np.array(data_input)
    y = np.array(data_target).reshape((-1, 1))

    return x, y

def phi(x, mu, sigma):

    l = len(mu)
    res = np.empty((x.shape[0], l))
    for i in range(l):
        arg = x - mu[i]
        sigma_inv = np.linalg.pinv(sigma[i])
        for j in range(x.shape[0]):
            tmp = arg[j, :].reshape((-1, 1))
            res[j, i] = np.exp(tmp.T.dot(sigma_inv).dot(tmp)*(-0.5))
    return res

def linear(x, w, mu, sigma):
    p = phi(x, mu, sigma)
    p = np.hstack((np.ones((p.shape[0], 1)), p))
    return p.dot(w)

def loss(y, p):
    tmp = y - p
    return tmp.T.dot(tmp) / 2

def reg(w, lamb):
    return lamb * w.T.dot(w)

def rmse(y, p, w, lamb, n):
    return np.sqrt(2 * (loss(y, p) + reg(w, lamb)) / n)[0, 0]

def mu_simple(x, l):
    idx = random.sample(list(range(0, x.shape[0])), l)
    mu = []
    for i in idx:
        mu.append(x[i, :])
    return mu

def preproses(x):
    var = np.var(x, axis=0)
    idx = var!=0
    return x[:, idx], idx

def sigma_simple(x, l):
    var = 1 + np.var(x, axis=0) / 10
    sigma1 = np.diag(var)
    return [sigma1] * l

def closed_form_sol(x, y, mu, sigma, lamb):
    Phi = phi(x, mu, sigma)
    Phi = np.hstack((np.ones((Phi.shape[0], 1)), Phi))
    tmp1 = np.diag(np.ones(len(mu) + 1) * lamb)
    tmp2 = Phi.T.dot(Phi)
    tmp = np.linalg.inv(tmp1 + tmp2)
    w_sol = tmp.dot(Phi.T).dot(y)
    return w_sol

def sgd(x, y, lamb, lr_low, lr_high, mu, sigma, epochs, mbatch_size):
    w = np.zeros((len(mu) + 1, 1))
    Phi = phi(x, mu, sigma)
    Phi = np.hstack((np.ones((Phi.shape[0], 1)), Phi))
    n = x.shape[0]
    batch_number = int(np.ceil(x.shape[0] / mbatch_size))
    for i in range(epochs):
        lr = lr_decay(lr_low, lr_high, n, i)
        for a in range(batch_number):
            tmp1 = Phi[a*mbatch_size:a*mbatch_size+mbatch_size, :].dot(w)
            tmp = Phi[a*mbatch_size:a*mbatch_size+mbatch_size, :].T.dot(y[a*mbatch_size:a*mbatch_size+mbatch_size, :] - tmp1) / mbatch_size
            loss_grad = (lamb*w - tmp)
            w -= lr*loss_grad
    return w

def lr_decay(low, high, n, t):
    lr = high + (low - high) * t / n
    return lr


if __name__ == '__main__':
    # Extract feature values and labels from the data
    file_path = r'path/to/Querylevelnorm.txt'
    x, y = import_data(file_path)
    #x = np.hstack((np.ones((x.shape[0], 1)), x))
    n, m = x.shape

    # Data partition
    n_train = int(0.8 * n)
    n_valid = int((n - n_train) / 2)

    x_train = x[:n_train, :]
    x_valid = x[n_train:n_train + n_valid, :]
    x_test = x[n_train + n_valid:, :]

    x_train, idx = preproses(x_train)
    x_valid = x_valid[:, idx]
    x_test = x_test[:, idx]

    y_train = y[:n_train, :]
    y_valid = y[n_train:n_train + n_valid, :]
    y_test = y[n_train + n_valid:, :]

    l = 10
    lamb = 1
    lr_low = 10e-17
    lr_high = 10e-10
    epochs = 50
    mbatch_size = 50

    mu = mu_simple(x_train, l)
    sigma = sigma_simple(x_train, l)

    # Random initialization
    w0 = np.random.uniform(0, 1, (len(mu)+1, 1))
    p0 = linear(x_test, w0, mu, sigma)
    rmse0 = rmse(y_test, p0, w0, lamb, y_test.shape[0])
    print("Error with random weights: " + str(rmse0))

    # Closed form solution
    w = closed_form_sol(x_train, y_train, mu, sigma, lamb)

    p = linear(x_test, w, mu, sigma)
    print(w)
    rmse0 = rmse(y_test, p, w, lamb, y_test.shape[0])
    print("Error using closed form solution: " + str(rmse0))

    #SGD
    w = sgd(x_train, y_train, lamb, lr_low, lr_high, mu, sigma, epochs, mbatch_size)
    p = linear(x_test, w, mu, sigma)
    print(w)
    rmse0 = rmse(y_test, p, w, lamb, y_test.shape[0])
    print("Error using SGD:" + str(rmse0))


