import time
import random

import numpy as np
import pypolyagamma as pypolyagamma


def calculate_D(S):
    N = S.shape[1]

    D = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.dot(S[1:, i].T, S[:-1, j])

    return D


def calculate_C_w(S, w_i):
    N = S.shape[1]
    T = S.shape[0]

    C_w = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            C_w[i, j] = 4. * np.dot(S[:, i].T, np.multiply(w_i, S[:, j]))

    return C_w


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

    ppg.pgdrawv(A, 2. * np.dot(S, J_i), w_i)

    return w_i


def sample_J_i(S, C, D_i, w_i, gamma_i, sigma_J):
    N = S.shape[1]
    J_i = np.zeros(N)

    included_ind = list(np.where(gamma_i > 0)[0])

    if len(included_ind) == 0:
        return J_i

    cov_mat = (1. / sigma_J) * np.identity(N)

    C_gamma = C[:, included_ind][included_ind, :]
    cov_mat_gamma = cov_mat[included_ind, :][:, included_ind]
    D_i_gamma = D_i[included_ind]

    mean = np.dot(np.linalg.inv(C_gamma + cov_mat_gamma), D_i_gamma)
    cov = C_gamma + cov_mat_gamma

    J_i_gamma = np.random.multivariate_normal(mean, cov)

    J_i[included_ind] = J_i_gamma

    return J_i


def calc_block_dets(C_gamma, j_rel, sigma_J, num_active):
    cov_mat = (1. / sigma_J) * np.identity(num_active)
    mat = cov_mat + C_gamma

    A = mat[:j_rel, :j_rel]

    B_1 = mat[:j_rel, j_rel:]
    C_1 = mat[j_rel:, :j_rel]
    D_1 = mat[j_rel:, j_rel:]

    B_0 = mat[:j_rel, j_rel + 1:]
    C_0 = mat[j_rel + 1:, :j_rel]
    D_0 = mat[j_rel + 1:, j_rel + 1:]

    det_cov_1 = float(num_active) * sigma_J
    det_cov_0 = float(num_active - 1) * sigma_J

    # import ipdb;ipdb.set_trace()
    # If the matrix is small don't bother to split
    if mat.shape[0] < 5.:
        pre_factor_1 = 1. / (det_cov_1 * np.linalg.det(mat))
        pre_factor_0 = 1. / (det_cov_0 * np.linalg.det(np.delete(np.delete(mat, j_rel, 0), j_rel, 1)))

    elif j_rel == 0:
        pre_factor_0 = 1. / (det_cov_0 * np.linalg.det(D_0))
        pre_factor_1 = 1. / (det_cov_1 * np.linalg.det(mat))
    elif j_rel == num_active - 1:
        pre_factor_0 = 1. / (det_cov_0 * np.linalg.det(A))
        pre_factor_1 = 1. / (det_cov_1 * np.linalg.det(mat))
    else:
        det_A = np.linalg.det(A)
        A_inv = np.linalg.inv(A)
        pre_factor_0 = 1. / (det_cov_0 * det_A * np.linalg.det(D_0 - np.dot(C_0, np.dot(A_inv, B_0))))
        pre_factor_1 = 1. / (det_cov_1 * det_A * np.linalg.det(D_1 - np.dot(C_1, np.dot(A_inv, B_1))))

    return np.sqrt(pre_factor_0), np.sqrt(pre_factor_1)


def calc_gamma_prob(sigma_J, C_gamma, D_i_gamma, ro, j_rel):
    num_active = D_i_gamma.shape[0]  # How manny gammas are equal to 1
    cov_mat = 1. / sigma_J * np.identity(num_active)
    mat = cov_mat + C_gamma
    mat_inv = np.linalg.inv(mat)

    # calculate determinant with and without j in block form
    prefactor_0, prefactor_1 = calc_block_dets(C_gamma, j_rel, sigma_J, num_active)

    D_i_gamma_0 = np.delete(D_i_gamma, j_rel)
    mat_0_inv = np.linalg.inv(np.delete(np.delete(mat, j_rel, 0), j_rel, 1))

    gauss_0 = np.exp(0.5 * np.dot(D_i_gamma_0.T, np.dot(mat_0_inv, D_i_gamma_0)))
    gauss_1 = np.exp(0.5 * np.dot(D_i_gamma.T, np.dot(mat_inv, D_i_gamma)))
    # import ipdb; ipdb.set_trace()
    prob_0 = prefactor_0 * gauss_0 * (1. - ro)
    prob_1 = prefactor_1 * gauss_1 * ro

    # TODO: How to handle infinity?
    if np.isinf(gauss_0):
        if np.isinf(gauss_1):
            return 0.5
        return 0.

    if np.isinf(gauss_1):
        return 1.

    new_ro = prob_1 / (prob_1 + prob_0)

    if np.isnan(new_ro):
        import ipdb;
        ipdb.set_trace()

    return new_ro


def sample_gamma_i(gamma_i, D_i, C, ro, sigmma_J):
    N = C.shape[0]

    for j in range(N):
        # import ipdb; ipdb.set_trace()
        gamma_i[j] = 1.
        active_indices = np.where(gamma_i > 0)[0]

        # Don't allow a network with 0 connections
        if len(active_indices) == 1.:
            continue

        j_rel = j - np.where(gamma_i[:j] == 0)[0].shape[0]
        D_i_gamma = D_i[active_indices]
        C_gamma = C[:, active_indices][active_indices, :]

        new_ro = calc_gamma_prob(sigmma_J, C_gamma, D_i_gamma, ro, j_rel)
        # import ipdb; ipdb.set_trace()
        try:
            gamma_i[j] = np.random.binomial(1, new_ro, 1)
        except ValueError:
            import ipdb;
            ipdb.set_trace()

    return gamma_i


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
    # import ipdb; ipdb.set_trace()
    samples_w_i = np.zeros((N_s, T), dtype=np.float32)
    samples_J_i = np.zeros((N_s, N), dtype=np.float32)
    samples_gamma_i = np.zeros((N_s, N), dtype=np.float32)

    gamma_i = np.random.binomial(1, ro, N)
    J_i = np.multiply(gamma_i, np.random.normal(0, sigma_J, N))

    for i in xrange(N_s):
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        J_i = sample_J_i(S, C_w_i, D_i, w_i, gamma_i, sigma_J)
        gamma_i = sample_gamma_i(gamma_i, D_i, C_w_i, ro, sigma_J)

        samples_w_i[i, :] = w_i
        samples_J_i[i, :] = J_i
        samples_gamma_i[i, :] = gamma_i

    return samples_w_i[burnin:, :], samples_J_i[burnin:, :], samples_gamma_i[burnin:, :]
    # else:
    #    return samples_w_i[burnin:N_s:thin, :], samples_J_i[burnin:N_s:thin, :], \
    #           samples_gamma_i[burnin:N_s:thin, :]
