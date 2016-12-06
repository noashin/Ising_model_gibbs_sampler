import numpy as np
import pypolyagamma as pypolyagamma


def calculate_D(S):
    N = S.shape[1]

    D = np.empty((N, N))

    for i in range(N):
        for j in range(N):
            D[i, j] = np.dot(S[1:, i].T, S[:-1, j])

    return 0.5 * D


def calculate_C_w(S, w_i):
    w_mat = np.diag(w_i)

    return np.dot(S.T, np.dot(w_mat, S))


def sample_w_i(S, J_i):
    """

    :param S: observation matrix
    :param J_i: neuron i's couplings
    :return: samples for w_i from a polyagamma distribution
    """

    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))

    T = S.shape[0]
    A = np.ones(T)
    w_i = np.zeros(T)

    ppg.pgdrawv(A, np.dot(S, J_i), w_i)

    return w_i


def sample_J_i(S, C, D_i, w_i, sigma_J):
    N = S.shape[1]

    cov_mat = (1. / sigma_J) * np.identity(N)

    mean = np.dot(C + cov_mat, D_i)
    cov = np.linalg.inv(C + cov_mat)

    J_i = np.random.multivariate_normal(mean, cov)

    return J_i


def sample_neuron(samp_num, burnin, sigma_J, S, D_i, ro, thin=0):
    """ This function uses the Gibbs sampler to sample from w, gamma and J

    :param samp_num: Number of samples to be drawn
    :param burnin: Number of samples to burn in
    :param sigma_J: variance of the J slab
    :param S: Neurons' activity matrix. Including S0. (T + 1) x N
    :param C: observation correlation matrix. N x N
    :param D_i: time delay correlations of neuron i. N
    :return: samp_num samples (each one of length K (time_steps)) from the posterior distribution for w,x,z.
    """

    # random.seed(seed)

    T, N = S.shape

    # actual number of samples needed with thining and burin-in
    if (thin != 0):
        N_s = samp_num * thin + burnin
    else:
        N_s = samp_num + burnin

    samples_w_i = np.zeros((N_s, T), dtype=np.float32)
    samples_J_i = np.zeros((N_s, N), dtype=np.float32)

    J_i = np.random.normal(0, sigma_J, N)

    for i in xrange(N_s):
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        J_i = sample_J_i(S, C_w_i, D_i, w_i, sigma_J)

        samples_w_i[i, :] = w_i
        samples_J_i[i, :] = J_i

    if thin == 0:
        samples_w_i[burnin:, :], samples_J_i[burnin:, :]
    else:
        return samples_w_i[burnin:N_s:thin, :], samples_J_i[burnin:N_s:thin, :]
