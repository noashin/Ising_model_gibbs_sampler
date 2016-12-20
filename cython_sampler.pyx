import numpy as np
cimport numpy as np
import pypolyagamma as pypolyagamma

DTYPE = np.float32
ctypedef float DTYPE_t
ctypedef double DTYPE_d


def calculate_D(np.ndarray[DTYPE_t, ndim=2] S):
    cdef int N
    cdef np.ndarray[DTYPE_t, ndim=2] D

    N = S.shape[1]

    D = np.zeros((N, N), dtype=DTYPE)

    for i in range(N):
        for j in range(N):
            D[i, j] = np.dot(S[1:, i].T, S[:-1, j])

    return D * 0.5


cdef np.ndarray[DTYPE_t, ndim=2] calculate_C_w(np.ndarray[DTYPE_t, ndim=2] S,
                        np.ndarray[DTYPE_t, ndim=1] w_i):
    cdef np.ndarray[DTYPE_t, ndim=2] w_mat

    w_mat = np.diag(w_i)

    return np.dot(S.T, np.dot(w_mat, S))


cdef np.ndarray[DTYPE_t, ndim=2] sample_w_i(np.ndarray[DTYPE_t, ndim=2] S,
                np.ndarray[DTYPE_t, ndim=1] J_i):
    """

    :param S: observation matrix
    :param J_i: neuron i's couplings
    :return: samples for w_i from a polyagamma distribution
    """

    cdef int T = S.shape[0]
    cdef np.ndarray[DTYPE_d, ndim=1] A = np.ones(T)
    cdef np.ndarray[DTYPE_d, ndim=1] w_i = np.zeros(T)

    cdef int nthreads = pypolyagamma.get_omp_num_threads()
    cdef np.ndarray[DTYPE_d, ndim=1] seeds = np.random.randint(2**16, size=nthreads).astype(np.float64)
    ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]

    pypolyagamma.pgdrawvpar(ppgs, A, np.dot(S, J_i).astype(np.float64), w_i)

    return w_i.astype(np.float32)


cdef np.ndarray[DTYPE_t, ndim=1] sample_J_i(np.ndarray[DTYPE_t, ndim=2] S,
                np.ndarray[DTYPE_t, ndim=2] C,
                np.ndarray[DTYPE_t, ndim=1] D_i,
                np.ndarray[DTYPE_t, ndim=1] w_i,
                np.ndarray[DTYPE_t, ndim=1] gamma_i,
                float sigma_J):

    cdef int N = S.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] J_i = np.zeros(N, dtype=DTYPE)

    cdef np.ndarray[int, ndim=1] included_ind = np.where(gamma_i > 0)[0].astype(np.int32)

    if included_ind.shape[0] == 0:
        return J_i

    cdef np.ndarray[DTYPE_t, ndim=2] cov_mat = (1. / sigma_J) * np.identity(N, dtype=np.float32)

    cdef np.ndarray[DTYPE_t, ndim=2] C_gamma = C[:, included_ind][included_ind, :]
    cdef np.ndarray[DTYPE_t, ndim=2] cov_mat_gamma = cov_mat[included_ind, :][:, included_ind]
    cdef np.ndarray[DTYPE_t, ndim=1] D_i_gamma = D_i[included_ind]

    cdef np.ndarray[DTYPE_t, ndim=2] cov = np.linalg.inv(C_gamma + cov_mat_gamma)
    cdef np.ndarray[DTYPE_t, ndim=1] mean = np.dot(cov, D_i_gamma)

    cdef np.ndarray[DTYPE_t, ndim=1] J_i_gamma = np.random.multivariate_normal(mean, cov).astype(np.float32)

    J_i[included_ind] = J_i_gamma

    return J_i


cdef np.ndarray[DTYPE_t, ndim=1] calc_block_dets(np.ndarray[DTYPE_t, ndim=2] C_gamma,
                    int j_rel,
                    float sigma_J,
                    int num_active):

    cdef np.ndarray[DTYPE_t, ndim=2] cov_mat = (1. / sigma_J) * np.identity(num_active, dtype=np.float32)
    cdef np.ndarray[DTYPE_t, ndim=2] mat = cov_mat + C_gamma
    cdef double max_mat = np.max(mat)
    mat = mat / max_mat

    cdef np.ndarray[DTYPE_t, ndim=2] A = mat[:j_rel, :j_rel]

    cdef np.ndarray[DTYPE_t, ndim=2] B_1 = mat[:j_rel, j_rel:]
    cdef np.ndarray[DTYPE_t, ndim=2] C_1 = mat[j_rel:, :j_rel]
    cdef np.ndarray[DTYPE_t, ndim=2] D_1 = mat[j_rel:, j_rel:]

    cdef np.ndarray[DTYPE_t, ndim=2] B_0 = mat[:j_rel, j_rel + 1:]
    cdef np.ndarray[DTYPE_t, ndim=2] C_0 = mat[j_rel + 1:, :j_rel]
    cdef np.ndarray[DTYPE_t, ndim=2] D_0 = mat[j_rel + 1:, j_rel + 1:]

    cdef float det_cov_1 = float(num_active) * sigma_J
    cdef float det_cov_0 = float(num_active - 1) * sigma_J

    cdef float pre_factor_0
    cdef float prefactor_1
    cdef float det_A
    cdef np.ndarray[DTYPE_t, ndim=2] A_inv

    # import ipdb;ipdb.set_trace()
    # If the matrix is small don't bother to split
    if mat.shape[0] < 5.:
        pre_factor_1 = (det_cov_1 / (np.linalg.det(mat) * max_mat))
        pre_factor_0 = (det_cov_0 / np.linalg.det(np.delete(np.delete(mat, j_rel, 0), j_rel, 1)))

    elif j_rel == 0:
        pre_factor_0 = det_cov_0 / np.linalg.det(D_0)
        pre_factor_1 = det_cov_1 / (np.linalg.det(mat) * max_mat)

    elif j_rel == num_active - 1:
        pre_factor_0 = det_cov_0 / np.linalg.det(A)
        pre_factor_1 = det_cov_1 / (np.linalg.det(mat) * max_mat)

    else:
        det_A = np.linalg.det(A)
        A_inv = np.linalg.inv(A).astype(np.float32)
        pre_factor_0 = det_cov_0 / (det_A * np.linalg.det(D_0 - np.dot(C_0, np.dot(A_inv, B_0))))
        pre_factor_1 = det_cov_1 / (max_mat * det_A * np.linalg.det(D_1 - np.dot(C_1, np.dot(A_inv, B_1))))

    cdef np.ndarray[DTYPE_t, ndim=1] res = np.array([np.sqrt(pre_factor_0), np.sqrt(pre_factor_1)]).astype(DTYPE)

    return res


cdef float calc_gamma_prob(float sigma_J,
                    np.ndarray[DTYPE_t, ndim=2] C_gamma,
                    np.ndarray[DTYPE_t, ndim=1] D_i_gamma,
                    float ro, int j_rel):

    cdef int num_active = D_i_gamma.shape[0]  # How manny gammas are equal to 1
    cdef np.ndarray[DTYPE_t, ndim=2] cov_mat = 1. / sigma_J * np.identity(num_active, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] mat = cov_mat + C_gamma
    cdef np.ndarray[DTYPE_t, ndim=2] mat_inv = np.linalg.inv(mat).astype(DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] mat_0_inv = np.linalg.inv(np.delete(np.delete(
                                                                mat, j_rel, 0), j_rel, 1)).astype(DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] D_i_gamma_0 = np.delete(D_i_gamma, j_rel)

    # calculate determinant with and without j in block form
    cdef float prefactor_0
    cdef float prefactor_1

    cdef np.ndarray[DTYPE_t, ndim=1] res = calc_block_dets(C_gamma, j_rel, sigma_J, num_active).astype(DTYPE)

    prefactor_0 = res[0]
    prefactor_1 = res[1]


    cdef double sq_1 = 0.5 * np.dot(D_i_gamma.T, np.dot(mat_inv, D_i_gamma))
    cdef double sq_0 = 0.5 * np.dot(D_i_gamma_0.T, np.dot(mat_0_inv, D_i_gamma_0))

    cdef double new_ro = 1. / (1. + np.exp(sq_0 - sq_1 + np.log(1. - ro) - np.log(ro) +
                               np.log(prefactor_0) - np.log(prefactor_1)))

    return new_ro


cdef np.ndarray[DTYPE_t, ndim=1] sample_gamma_i(np.ndarray[DTYPE_t, ndim=1] gamma_i,
                    np.ndarray[DTYPE_t, ndim=1] D_i,
                    np.ndarray[DTYPE_t, ndim=2] C,
                    float ro, float sigmma_J):

    cdef int N = C.shape[0]
    cdef np.ndarray[int, ndim=1] active_indices
    cdef int j_rel
    cdef np.ndarray[DTYPE_t, ndim=1] D_i_gamma
    cdef np.ndarray[DTYPE_t, ndim=2] C_gamma
    cdef float new_ro

    for j in range(N):
        gamma_i[j] = 1.
        active_indices = np.where(gamma_i > 0)[0].astype(np.int32)

        # Don't allow a network with 0 connections
        if len(active_indices) == 1.:
            continue

        j_rel = j - np.where(gamma_i[:j] == 0)[0].shape[0]
        D_i_gamma = D_i[active_indices]
        C_gamma = C[:, active_indices][active_indices, :]

        new_ro = calc_gamma_prob(sigmma_J, C_gamma, D_i_gamma, ro, j_rel)
        gamma_i[j] = np.random.binomial(1, new_ro, 1)

    return gamma_i


cdef np.ndarray[DTYPE_t, ndim=3] sample_neuron_cython(int samp_num, int burnin, float sigma_J,
                    np.ndarray[DTYPE_t, ndim=2] S,
                    np.ndarray[DTYPE_t, ndim=1] D_i,
                    float ro, int thin=0):
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

    cdef int T = S.shape[0]
    cdef int N = S.shape[1]

    cdef int N_s

    # actual number of samples needed with thining and burin-in
    if (thin != 0):
        N_s = samp_num * thin + burnin
    else:
        N_s = samp_num + burnin

    cdef np.ndarray[DTYPE_t, ndim=1] gamma_i = np.ones(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] J_i = np.multiply(gamma_i, np.random.normal(0, sigma_J, N).astype(DTYPE))

    cdef np.ndarray[DTYPE_t, ndim=1] w_i
    cdef np.ndarray[DTYPE_t, ndim=2] C_w_i

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.empty((3, N_s, T), dtype=DTYPE)

    for i in xrange(N_s):
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        gamma_i = sample_gamma_i(gamma_i, D_i, C_w_i, ro, sigma_J)
        J_i = sample_J_i(S, C_w_i, D_i, w_i, gamma_i, sigma_J)

        res[0, :, :] = w_i
        res[1, :, :N] = J_i
        res[2, :, :N] = gamma_i

    if thin == 0:
        return res[:, burnin:, :]
    else:
        return res[:, burnin:N_s:thin, :]


cdef np.ndarray[DTYPE_t, ndim=3] sample_neuron_cython_save_sufficient(int samp_num, int burnin, float sigma_J,
                    np.ndarray[DTYPE_t, ndim=2] S,
                    np.ndarray[DTYPE_t, ndim=1] D_i,
                    float ro, int thin=0):
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

    cdef int T = S.shape[0]
    cdef int N = S.shape[1]

    cdef int N_s

    # actual number of samples needed with thining and burin-in
    if (thin != 0):
        N_s = samp_num * thin + burnin
    else:
        N_s = samp_num + burnin

    cdef np.ndarray[DTYPE_t, ndim=1] gamma_i = np.ones(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] J_i = np.multiply(gamma_i, np.random.normal(0, sigma_J, N).astype(DTYPE))

    cdef np.ndarray[DTYPE_t, ndim=1] w_i
    cdef np.ndarray[DTYPE_t, ndim=2] C_w_i

    cdef np.ndarray[DTYPE_t, ndim=3] res = np.zeros((2, 3, T), dtype=DTYPE)

    for i in xrange(N_s):
        # import ipdb; ipdb.set_trace()
        w_i = sample_w_i(S, J_i)
        C_w_i = calculate_C_w(S, w_i)
        gamma_i = sample_gamma_i(gamma_i, D_i, C_w_i, ro, sigma_J)
        J_i = sample_J_i(S, C_w_i, D_i, w_i, gamma_i, sigma_J)

        if N_s > burnin:
            res[0, 0, :] += w_i
            res[0, 1, :N] += J_i
            res[0, 2, :N] += gamma_i

            res[1, 0, :] += np.power(w_i, 2)
            res[1, 1, :N] += np.power(J_i, 2)
            res[1, 2, :N] += np.power(gamma_i, 2)

    res[:, :, :] = res[:, :, :] / float(samp_num)

    res[1, :, :] -= np.power(res[0, :, :], 2)

    return res


def sample_neuron(int samp_num, int burnin, float sigma_J,
                    np.ndarray[DTYPE_t, ndim=2] S,
                    np.ndarray[DTYPE_t, ndim=1] D_i,
                    float ro, int thin=0,
                    bint save_all=True):

    cdef int N = S.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] res

    if save_all:
        res = sample_neuron_cython(samp_num, burnin, sigma_J, S, D_i, ro, thin)
    else:
        res = sample_neuron_cython_save_sufficient(samp_num, burnin, sigma_J, S, D_i, ro, thin)

    return res


