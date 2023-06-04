import numpy as np
from .dnn_class import NeuralNetwork as nn
from . import multiplication as mult

def phi_1_k(k):
    # returns \Phi^{1,k} from the proof of Lemma 2.7 as DNN
    result = nn(2**(k-1)+2,[],2**(k+1)+2)
    for m in range(2**(k+1)+2):
        for i in range(2**(k-1)+2):
            if m == 0 and i == 0:
                result.weights[0][m][i] = 1
            elif m == 1 and i == 2**(k-1)+1:
                result.weights[0][m][i] = 1
            elif i == int(np.ceil((m+6)*.25))-1 and m > 1:
                result.weights[0][m][i] = 1
    return result

def phi(k):
    # returns \Phi^{\prime} from the proof of Lemma 2.7 as DNN
    result = nn(2**k+2,[],2**k+2)
    for m in range(2**k+2):
        for i in range(2**k+2):
            if m == i and i <= 1:
                result.weights[0][m][i] = 1
            elif m == i and i >= 2:
                result.weights[0][m][i] = 2
            elif i == 0 and m >= 2 and m%2 == 0:
                result.weights[0][m][i] = -1
        if m >= 2 and m%2 != 0:
            result.biases[0][m] = -1
    return result

def phi_delta_2_k(k, delta):
    # returns \Phi_\delta^{2,k} from the proof of Lemma 2.7 as DNN
    L_delta, _ = mult.mult_dnn(2, delta, 2).count_depth_size()
    prods = []
    for _ in range(2**k):
        prods.append(mult.mult_dnn(2, delta, 2))
    first_part = phi(k)
    second_part = mult.identity_dnn_n(2, L_delta).n_parallelize(prods)
    return first_part.concatenate(second_part)
    
def psi_delta_k(k, delta=.001):
    # returns \Psi_\delta^k from the proof of Lemma 2.7 as DNN
    if k == 1:
        base_mult = mult.mult_dnn(2, delta*.25)
        l_1, _ = base_mult.count_depth_size()
        a = np.ones((2, 1))
        b_prime = [-1]
        A_i, b_i = base_mult.weights, base_mult.biases
        phi = mult.mult_dnn(2, delta*.25)
        phi.weights[0] = np.matmul(A_i[0], a)
        phi.weights[l_1] = 2*A_i[l_1]
        phi.biases[l_1] = 2*b_i[l_1] + b_prime
        phi.input_dim = 1
        id_a = mult.identity_dnn_n(1, l_1)
        id_b = mult.identity_dnn_n(1, l_1)
        psi = id_a.n_undistinct_parallelize([id_b, phi])
        return psi
    else:
        theta = 2**(-2*k-4)*delta
        phi_psi = phi_1_k(k-1).concatenate(psi_delta_k(k-1, theta))
        psi_delta_k_= phi_delta_2_k(k-1,delta).concatenate(phi_psi)
        return psi_delta_k_

def phi_3_n(k,n):
    # returns \Phi^{3,n} from the proof of Lemma 2.7 as DNN
    result = nn(2**k+2*k-1,[],n)
    result.weights[0][0][1] = 1
    result.weights[0][1][2] = 1
    for l in range(2,n):
        j = int(np.ceil(np.log2(l+1)))
        result.weights[0][l][l+2*j-1] = 1
    return result

def cheb_delta_n(delta, n):
    # returns \Phi_\delta^{Cheb,n} from the proof of Lemma 2.7 as DNN
    if n == 1:
        result = nn(1,[],1)
        result.weights[0][0][0] = 1
        return result
    k = int(np.ceil(np.log2(n)))
    psi_delta_k_ = psi_delta_k(k, delta)
    depth_psi, _ = psi_delta_k_.count_depth_size()
    depth_psi_delta_k_plus_one = depth_psi + 1
    big_parallelization = []
    for j in range(k):
        psi_j = psi_delta_k(j+1, delta)
        depth_psi_j, _ = psi_j.count_depth_size()
        l_j = depth_psi_delta_k_plus_one - depth_psi_j
        element = psi_j.concatenate(mult.identity_dnn_n(1, l_j))
        big_parallelization.append(element)
    second_part = big_parallelization[0].n_undistinct_parallelize(big_parallelization[1:])
    return phi_3_n(k,n).concatenate(second_part)

def m_infty(vectors):
    # returns the maximum absolute value of the entries of the vectors
    max_deg = 0
    for vector in vectors:
        for entry in vector:
            if np.absolute(entry) > max_deg:
                max_deg = entry
    return max_deg

def phi_Lambda_delta_2(k, var_lambda, delta =.001):
    # returns \Phi_{\Lambda,\delta}^{(2)} from the proof of Theorem 2.8 as DNN
    m = m_infty(var_lambda)
    beta = .5*delta*((1+delta)**(-k))*(k**(-1))*(m +1)**(-1)
    base = cheb_delta_n(beta, m)
    cheb_list =[]
    for i in range(k):
        cheb_list.append(base)
    if len(cheb_list) == 1:
        return cheb_list[0]
    else:
        return cheb_list[0].n_parallelize(cheb_list[1:])

def reorder_output(k, var_lambda):
    # returns the reordered output of \Phi_{\Lambda,\delta}^{(2)} from the proof of Theorem 2.8 as DNN
    m = m_infty(var_lambda)
    reorder = nn(k*m,[],len(var_lambda)*k)
    counter = 0
    for vector in var_lambda:
        for j in range(len(vector)):
            if vector[j] == 0:
                counter += 1
            else:
                reorder.weights[0][counter][vector[j]-1+j*m] = 1    
                counter += 1
    # add bias for Chebyshev polynomials of degree 0
    for i in range(len(reorder.weights[0])):    
        if sum(reorder.weights[0][i]) == 0:
            reorder.biases[0][i] = 1
    return reorder

def phi_Lambda_delta_1(k, var_lambda, delta =.001):
    # returns \Phi_{\Lambda,\delta}^{(1)} from the proof of Theorem 2.8 as DNN
    m = m_infty(var_lambda)
    beta_prime = .5*delta*(m**2 +1)**(-1)
    base = mult.mult_dnn(k, beta_prime, 1+delta)
    res = []
    for _ in var_lambda:
        res.append(base)
    return res[0].n_parallelize(res[1:])

def phi_Lambda_delta(k, var_lambda, delta =.001):
    # returns \Phi_{\Lambda,\delta} from the proof of Theorem 2.8 as DNN
    phi_2 = phi_Lambda_delta_2(k, var_lambda, delta)
    ordered_phi_2 = reorder_output(k, var_lambda).concatenate(phi_2)
    return phi_Lambda_delta_1(k, var_lambda, delta).concatenate(ordered_phi_2)
