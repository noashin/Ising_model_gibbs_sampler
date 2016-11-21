import numpy as np


def test_calc_block_dets_normal():
    from sampler import calc_block_dets
    num_active = 8
    tmp = np.random.randn(num_active, num_active)
    C_gamma = np.dot(tmp.T, tmp)
    j_rel = 3
    sigma_J = 0.5
    cov_mat = sigma_J * np.identity(num_active)
    cov_mat_inv = (1. / sigma_J) * np.identity(num_active)

    cov_mat_0 = np.delete(np.delete(cov_mat, j_rel, 0), j_rel, 1)
    cov_mat_inv_0 = np.linalg.inv(cov_mat_0)
    C_gamma_0 = np.delete(np.delete(C_gamma, j_rel, 0), j_rel, 1)

    prefactor_1 = np.linalg.det(np.linalg.inv(cov_mat_inv + C_gamma)) / (sigma_J * num_active)
    prefactor_0 = np.linalg.det(np.linalg.inv(cov_mat_inv_0 + C_gamma_0)) / (sigma_J * (num_active - 1))

    res = calc_block_dets(C_gamma, j_rel, sigma_J, num_active)

    np.testing.assert_array_almost_equal(prefactor_0, res[0])
    np.testing.assert_array_almost_equal(prefactor_1, res[1])


def test_calc_block_dets_edges():
    from sampler import calc_block_dets
    num_active = 8
    tmp = np.random.randn(num_active, num_active)
    C_gamma = np.dot(tmp.T, tmp)
    j_rel = 0
    sigma_J = 0.5
    cov_mat = sigma_J * np.identity(num_active)
    cov_mat_inv = (1. / sigma_J) * np.identity(num_active)

    cov_mat_0 = np.delete(np.delete(cov_mat, j_rel, 0), j_rel, 1)
    cov_mat_inv_0 = np.linalg.inv(cov_mat_0)
    C_gamma_0 = np.delete(np.delete(C_gamma, j_rel, 0), j_rel, 1)

    prefactor_1 = np.linalg.det(np.linalg.inv(cov_mat_inv + C_gamma)) / (sigma_J * num_active)
    prefactor_0 = np.linalg.det(np.linalg.inv(cov_mat_inv_0 + C_gamma_0)) / (sigma_J * (num_active - 1))

    res = calc_block_dets(C_gamma, j_rel, sigma_J, num_active)

    np.testing.assert_array_almost_equal(prefactor_0, res[0])
    np.testing.assert_array_almost_equal(prefactor_1, res[1])


def test_calc_gamma_prob():
    from sampler import calc_block_dets, calc_gamma_prob
    num_active = 2
    sigma_J = 0.5
    j_rel = 0
    ro = 0.5
    C_gamma = np.array([[1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.]])
    D_i_gamma = np.array([1., 0., 1., 1.])

    prefactor_0, prefactor_1 = calc_block_dets(C_gamma, j_rel, sigma_J, 4)

    prob_0 = prefactor_0 * np.exp(8.) * 0.7
    prob_1 = prefactor_1 * np.exp(15.) * 0.3

    new_ro = prob_1 / (prob_0 + prob_1)

    np.testing.assert_array_almost_equal(calc_gamma_prob(sigma_J, C_gamma, D_i_gamma, ro, j_rel), new_ro)
