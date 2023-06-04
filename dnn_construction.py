import numpy as np
from utils.helper_functions import cos_nd, generate_combinations
from utils.chebyshev import Chebyshev
from utils.cheb_dnn import phi_Lambda_delta
import utils.dnn_class as nn

def compute_dnn(k, eval_f, epsilon, alpha, delta_0, n_0):
    # k: input dimension
    # eval_f: function to approximate
    # epsilon: error tolerance
    # alpha: parameter for error tolerance
    # delta_0: initial delta
    # n_0: initial n
    # returns DNN approximating eval_f with error tolerance epsilon as described in Chapter 3, depth of the DNN, size of the DNN, n, delta and the error in the L_\infty norm
    n = n_0
    delta = delta_0
    state = True
    err = 1
    iter = 1
    while err > epsilon:
        var_lambda = generate_combinations(n, k)      
        err_1 = err
        chebyshev = Chebyshev(eval_f,n,k)
        _, coeffs = chebyshev.coefficients()
        phi_l_d = phi_Lambda_delta(k, var_lambda, delta)
        nn_coeffs = nn.NeuralNetwork(len(var_lambda),[],1)
        for i in range(len(var_lambda)):
            nn_coeffs.weights[0][0][i] = coeffs[i]
        phi = nn_coeffs.concatenate(phi_l_d)
        xx = cos_nd(generate_combinations(n-1, k), n-1)
        f_interp = chebyshev.interpolate(xx, coeffs)
        f_realized = [phi.realize(x) for x in xx]
        err = np.max(np.absolute(f_interp-f_realized))
        if err > alpha*err_1:
            state = not state
        if state:
            n = n+1
        else:
            delta = .5*delta
        print(f'Iteration: {iter} Error: {err}, n: {n}, delta: {delta}')
        iter += 1
    if state:
        n_star = n-1
        delta_star = delta
    else:
        n_star = n
        delta_star = 2*delta
    var_lambda = generate_combinations(n_star, k)
    chebyshev = Chebyshev(eval_f,n_star,k)
    _, coeffs = chebyshev.coefficients()
    phi_l_d = phi_Lambda_delta(k, var_lambda, delta_star)
    nn_coeffs = nn.NeuralNetwork(len(var_lambda),[],1)
    for i in range(len(var_lambda)):
        nn_coeffs.weights[0][0][i] = coeffs[i]
    phi = nn_coeffs.concatenate(phi_l_d)
    xx = cos_nd(generate_combinations(n_star-1, k), n_star-1)
    f_interp = chebyshev.interpolate(xx, coeffs)
    f_realized = [phi.realize(x) for x in xx]
    err = np.max(np.absolute(f_interp-f_realized))  
    depth, size = phi.count_depth_size()
    print(f"Total depth: {depth}, Total size: {size}, n: {n_star}, delta: {delta_star}, error: {err}. Reached after {iter} iterations.")
    return phi, depth, size, n_star, delta_star, err