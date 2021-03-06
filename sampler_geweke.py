from __future__ import division

import time
import random

import os
import numpy as np
import pypolyagamma as pypolyagamma


def calculate_C_w(S, w_i):
    #w_mat = np.diag(w_i)
    sw = w_i[:, np.newaxis] * S

    return np.dot(S.T, sw)


def sample_w_i(S, J_i):
    """

    :param S: observation matrix
    :param J_i: neuron i's couplings
    :return: samples for w_i from a polyagamma distribution
    """
    nthreads = pypolyagamma.get_omp_num_threads()
    seeds = np.random.randint(2 ** 16, size=nthreads)
    ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]

    T = S.shape[0]
    A = np.ones(T)
    w_i = np.zeros(T)
    pypolyagamma.pgdrawvpar(ppgs, A, np.dot(S, J_i), w_i)
    return w_i


def sample_J_gamma_i(S, C_w_i, D_i, sigma_J, J_i, ro, n, bias, bias_mean):
    N = S.shape[1]

    sigma_J_inv_mat = np.linalg.inv(sigma_J * np.identity(N))
    cov = C_w_i + sigma_J_inv_mat
    cov_inv = np.linalg.inv(cov)
    m = np.zeros(D_i.shape)
    m[-1] = bias_mean
    mu = np.dot(cov_inv, D_i + np.dot(sigma_J_inv_mat, m))

    gamma_i = np.zeros(N)

    for j in range(N):
        # import ipdb; ipdb.set_trace()
        v_j = cov[j]
        v_jj = cov[j, j]
        mu_j = mu[j]
        J_ij = J_i[j]

        alpha = np.dot(v_j, (J_i - mu)) - v_jj * (J_ij - mu_j)

        q_0_fac = mu_j * alpha - mu_j ** 2 * v_jj
        q_1_fac = alpha ** 2 / (4 * v_jj)  # * new_var / 2.

        BF = np.exp(q_0_fac - q_1_fac) / np.sqrt(np.pi / v_jj)

        if j == N - 1 and bias:
            gamma_ij = 1
        else:
            prob_1 = ro / (ro + BF * (1. - ro))
            try:
                gamma_ij = np.random.binomial(1, prob_1, 1)
            except ValueError:
                import ipdb;
                ipdb.set_trace()

        gamma_i[j] = gamma_ij

        if gamma_ij == 0:
            J_i[j] = 0
        else:
            try:
                J_i[j] = np.random.normal(mu_j - alpha / (2 * v_jj), np.sqrt(1. / (2. * v_jj)))
            except ValueError:
                import ipdb; ipdb.set_trace()
    return J_i, gamma_i


def sample_neuron_save_all(samp_num, burnin, sigma_J, S, D_i, ro, thin=0, save_all=True):
    print thin
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
    samples_gamma_i = np.zeros((N_s, N), dtype=np.float32)

    J_i = np.random.normal(0, sigma_J, N)

    for i in xrange(N_s):
        # print i
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        J_i, gamma_i = sample_J_gamma_i(S, C_w_i, D_i, sigma_J, J_i, ro)

        samples_w_i[i, :] = w_i
        samples_J_i[i, :] = J_i
        samples_gamma_i[i, :] = gamma_i

    if thin == 0:
        return samples_w_i[burnin:, :], samples_J_i[burnin:, :]
    else:
        return samples_w_i[burnin:N_s:thin, :], samples_J_i[burnin:N_s:thin, :]


def sample_neuron_save_sufficient(samp_num, burnin, sigma_J, S, D_i, ro, thin, n, bias, bias_mean):
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

    J_i = np.random.normal(0, sigma_J, N)
    if bias:
        J_i[-1] = np.random.normal(bias_mean, sigma_J)
    res = np.zeros((2, 3, T))
    # import ipdb; ipdb.set_trace()
    for i in xrange(N_s):
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        J_i, gamma_i = sample_J_gamma_i(S, C_w_i, D_i, sigma_J, J_i, ro, n, bias, bias_mean)

        if i > burnin:
            if (thin > 0 and i % thin == 0) or (thin == 0):
                res[0, 0, :] += w_i
                res[0, 1, :N] += J_i
                res[0, 2, :N] += gamma_i

                res[1, 0, :] += np.power(w_i, 2)
                res[1, 1, :N] += np.power(J_i, 2)
                res[1, 2, :N] += gamma_i

        if i % 1000 == 0:
            print i

    res[:, :, :] = res[:, :, :] / float(samp_num)

    res[1, :, :] -= np.power(res[0, :, :], 2)

    return res


def sample_neuron(samp_num, burnin, sigma_J, S, D_i, ro, thin, n, bias, bias_mean, save_all=True):
    # First - reseed!!
    np.random.seed()
    #import ipdb; ipdb.set_trace()
    if save_all:
        # TODO: Add bias!
        res = sample_neuron_save_all(samp_num, burnin, sigma_J, S, D_i, ro, thin)
    else:
        res = sample_neuron_save_sufficient(samp_num, burnin, sigma_J, S, D_i, ro, thin, n, bias, bias_mean)

    return res
