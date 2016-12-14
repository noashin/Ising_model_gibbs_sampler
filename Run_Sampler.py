# import pyximport
# pyximport.install()


import numpy as np
import multiprocessing as multiprocess
import click
import pickle
import time
import os

from cython_sampler import sample_neuron, calculate_D
from network_simulator import spike_and_slab, generate_spikes


def sample_neurons(samp_num, burnin, sigma_J, S, D_is, ro, input_indices, dir_name, thin=0):
    file_name = '_'.join(str(n) for n in input_indices)
    results = [sample_neuron(samp_num, burnin, sigma_J, S, D_is[n], ro, thin) for n in input_indices]

    with open(os.path.join(dir_name, file_name), 'wb') as f:
        pickle.dump(results, f)

    return results


def multi_process_sampling(args):
    return sample_neurons(*args)


def do_multiprocess(function_args, num_processes):
    """ processes the_args
        :param function:
        :param function_args:
        :param num_processes: how many pararell processes we want to run.
    """
    if num_processes > 1:
        pool = multiprocess.Pool(processes=num_processes)
        results_list = pool.map(multi_process_sampling, function_args)
        pool.close()
        pool.join()
    else:
        results_list = [sample_neurons(*some_args) for some_args in function_args]
    return results_list


def generate_J_S(bias, num_neurons, time_steps, sparsity, sigma_J):
    if bias != 0 and bias != 1:
        raise ValueError('bias should be either 1 or 0')

    N = num_neurons
    T = time_steps

    # Add a column for bias if it is part of the model
    J = spike_and_slab(sparsity, N, bias, sigma_J)
    J += 0.0
    S0 = - np.ones(N + bias)

    S = generate_spikes(N, T, S0, J, bias)

    return S, J


def do_inference(S, J, num_processes, samp_num, burnin, sigma_J, sparsity, dir_name, thin=0):
    T = S.shape[0]
    N = S.shape[1]
    D = calculate_D(S)

    # J_samps = np.empty((samp_num, N, N))
    # gamma_samps = np.empty((samp_num, N, N))
    # w_samps = np.empty((samp_num, T, N))

    # prepare inputs for multi processing
    args_multi = []
    indices = range(N)
    inputs = [indices[i:i + N / num_processes] for i in range(0, len(indices), N / num_processes)]
    for input_indices in inputs:
        args_multi.append((samp_num, burnin, sigma_J, S, D, sparsity, input_indices, dir_name, thin))

    results = do_multiprocess(args_multi, num_processes)

    # i = 0
    # for result in results:
    #    for neuron in result:
    #        w_samps[:, :, i] = neuron[0]
    #        J_samps[:, i, :] = neuron[1]
    #        gamma_samps[:, i, :] = neuron[2]

    return  # w_samps, J_samps, gamma_samps


@click.command()
@click.option('--num_neurons',
              help='number of neurons in the network. '
                   'If a list, the inference will be done for every number of neurons.')
@click.option('--time_steps',
              help='Number of time stamps. Length of recording. '
                   'If a list, the inference will be done for every number of time steps.')
@click.option('--num_processes', type=click.INT,
              default=1)
@click.option('--likelihood_function', type=click.STRING,
              default='probit',
              help='Should be either probit or logistic')
@click.option('--sparsity', type=click.FLOAT,
              default=0.3,
              help='Set sparsity of connectivity, aka ro parameter.')
@click.option('--pprior',
              help='Set pprior for the EP. If a list the inference will be done for every pprior')
@click.option('--activity_mat_file', type=click.STRING,
              default='')
@click.option('--bias', type=click.INT,
              default=0,
              help='1 if bias should be included in the model, 0 otherwise')
@click.option('--num_trials', type=click.INT,
              default=1,
              help='number of trials with different S ad J for given settings')
def main(num_neurons, time_steps, num_processes, likelihood_function, sparsity, pprior,
         activity_mat_file, bias, num_trials):
    N = 10
    T = 1000
    ro = 0.5
    sigma_J = 1. / N
    num_processes = 1
    samp_num = 1000

    burnin = 300
    thin = 0

    dir_name = './%s_%s_%s_%s_%s_%s_%s' % (time.strftime("%Y%m%d-%H%M%S"), N, T, ro, samp_num, thin, sigma_J)

    print dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    S, J = generate_J_S(0, N, T, ro, sigma_J)

    with open(os.path.join(dir_name, 'S_J'), 'wb') as f:
        pickle.dump([J, S], f)

    do_inference(S[1:, :], J, num_processes, samp_num, burnin, sigma_J, sparsity, dir_name)

    '''
    # If not generate S and J
    else:
        num_neurons = [int(num) for num in num_neurons.split(',')]
        time_steps = [int(num) for num in time_steps.split(',')]
        for i in range(num_trials):
            for N in num_neurons:
                for T in time_steps:
                    v_s = 1 #/ np.sqrt(N)
                    dir_name = get_dir_name(ppriors, N, T, sparsity, likelihood_function)
                    S, J, cdf_factor = generate_J_S(likelihood_function, bias, N, T, sparsity, v_s)
                    J_est_EPs = []
                    log_evidences = []
                    for pprior in ppriors:
                        results = do_inference(S, J, N, num_processes, pprior, cdf_factor, v_s)
                        J_est_EPs.append(results[0])
                        log_evidences.append(results[1])
                    save_inference_results_to_file(dir_name, S, J, bias, J_est_EPs, likelihood_function,
                                                   ppriors, log_evidences, [], i)'''


if __name__ == "__main__":
    main()
