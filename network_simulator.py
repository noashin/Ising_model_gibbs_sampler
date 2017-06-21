import numpy as np
from scipy import stats


def exp_cosh(H, beta=1.0):
    return 0.5 * np.exp(beta * H) / np.cosh(beta * H)


def kinetic_ising_model(S, J):
    """ Returns probabilities of S[t+1,:] being one.

    :param S: numpy.ndarray (T,N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence').
    :param J: numpy.ndarray (N, N)
        Coupling matrix

    :return: numpy.ndarray (T,N)
        Probabilities that at time point t+1 neuron n fires
    """
    # compute fields
    H = np.dot(S, J.T)
    # compute probabilities
    p = exp_cosh(H)
    # return
    return p


def spike_and_slab(ro, N, bias=0, v_s=1.0, bias_mean=0):
    ''' This function generate spike and priors

    :param ro: sparsity
    :param N: number of neurons
    :param bias: 1 if bias is included in the model, 0 other wise
    :return:
    '''

    gamma = stats.bernoulli.rvs(p=ro, size=(N, N + bias))
    normal_dist = np.random.normal(0.0, v_s, (N, N + bias))

    if bias:
        gamma[:, N] = 1
        normal_dist[:, N] = np.random.normal(bias_mean, v_s, N)

    return gamma * normal_dist


def generate_spikes(N, T, S0, J, bias=False, bias_mean=0, no_spike=-1, save=False):
    """ Generates spike data according to kinetic Ising model
        with a spike and slab prior.

    :param J: numpy.ndarray (N, N)
        Coupling matrix.
    :param T: int
        Length of trajectory that is generated.
    :param S0: numpy.ndarray (N)
        Initial pattern that sampling started from.
    :param bias: 1 if bias is included in the model. 0 other wise.
    :param no_spike: what number should represent 'no_spike'. Default is -1.

    :return: numpy.ndarray (T, N)
        Binary data where an entry is either 1 ('spike') or -1 ('silence'). First row is only ones for external fields.
    """

    # Initialize array for data
    S = np.empty([T, N + bias])
    # Set initial spike pattern
    S[0] = S0 if no_spike == -1 else np.zeros(N + bias)
    # Last column in the activity matrix is of the bias and should be 1 at all times
    if bias:
        S[:, N] = 1
    # Generate random numbers
    X = np.random.rand(T - 1, N)

    # Iterate through all time points
    for t in range(1, T):
        # Compute probabilities of neuron firing
        p = kinetic_ising_model(np.array([S[t - 1]]), J)
        # Check if spike or not
        if no_spike == -1:
            S[t, :N] = 2 * (X[t - 1] < p) - 1
        else:
            S[t, :N] = 2 * (X[t - 1] < p) / 2.0
    S = S
    return S
