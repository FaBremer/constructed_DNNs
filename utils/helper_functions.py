import numpy as np
import itertools
import random

def indicator(k: list, n: int) -> int:
     kk = [l for l in k]
     out = []
     for l in kk:
         S = sum(1 for i in l if i != 0 and i != n)
         out.append(S)
     return out

def generate_grid_points(n, k):
    # n: degree, i.e. maximum integer in grid
    # k: dimension
    # returns all possible combinations of integers from 0 to n in dimension k
    j = tuple(range(n+1))  # creates a tuple with integers from 0 to n inclusive
    possible_arrays = list(itertools.product(j, repeat=k))
    return np.array([list(p) for p in possible_arrays])

def generate_combinations(n, k):
    #print(f"Length should be {(n+1)**k}")
    j = tuple(range(n+1))  # creates a tuple with integers from 0 to n inclusive
    possible_arrays = list(itertools.product(j, repeat=k))
    return possible_arrays

def generate_random_points(n, k):
    points = []
    for _ in range(n ** k):
        point = [random.uniform(-1, 1) for _ in range(k)]
        points.append(point)
    return points

def cos_nd(grid, n):
    points = np.array(grid)
    x = points[:, 0]
    z = np.cos(x*np.pi/n) # first dimension
    for i in range(1, points.shape[1]):
        x = points[:, i]
        z = np.append(z,np.cos(x*np.pi/n)) # other dimensions
    return z.reshape(points.shape[1],points.shape[0]).T

"""
Functions below from prior work, not needed anymore, but left in for reference
"""
def sum_mult(f_hat,grid_cos):
     
    out = []
    for kj in grid_cos:
        prd = f_hat*kj
        #print(prd)
        out.append(np.sum(prd,axis=0))
         
    return np.asarray(out)

def cos_kj(grid_k,grid_j,n):
    kk = [kk for kk in grid_k]
    i = len(kk)
    #print(i)
    res = []
    for k in kk:
        #print(k)
        # TODO: Multiplication here contradictory to literature!!
        inner = np.prod(cos_nd(grid_j*k,n),axis=1)
        res.append(inner)
    return np.array(res)

def f_breve(k, n, f):
    import warnings
    warnings.warn("Use Chebychev class instead!",DeprecationWarning)
    # Generate all possible vectors x_j^n
    
    grid_k = generate_grid_points(n,k)
    #print(f"Shape of grid_k: {grid_k.shape}")
    grid_j = generate_grid_points(2*n-1,k)
    #print(f"Shape of grid_j: {grid_j.shape}")
    # Compute the indicator function 1_{S(n)}(k)
    #ind = indicator(k,n)
    
    grid_cos_j = cos_nd(grid_j,n)
    #print(f"Shape of grid_cos_j: {grid_cos_j.shape}")
    # f_hat = func_nd(grid_cos_j,f,n)
    f_hat = f(grid_cos_j.T)
    #print(f"Shape of f_hat: {f_hat.shape}")

    grid_cos_kj = cos_kj(grid_k,grid_j,n)
    #print(f"Shape of grid_cos_kj: {grid_cos_kj.shape}")
    sum_j = sum_mult(f_hat,grid_cos_kj)
    #print(f"Shape of sum_j: {sum_j.shape}")

    ind = indicator(grid_k,n)
    ind = np.array([2**i for i in ind])
    ret = ind*sum_j*(2*n)**(-k)

    return grid_k.reshape([n+1 for i in range(k)] + [k]), ret.reshape([n+1 for i in range(k)])
    #return grid_k, ret


def coeff_f_k_n(eval_f, k, n):
    return

# Get Polynom for k at multi-dimensional point x,y,...(args)
def chebychev_polynomials(k, *args):
    if k == 0:
        return 1
    elif k == 1:
        return np.prod(args)
    else:
        return 2*np.prod(args)*chebychev_polynomials(k-1, *args) - chebychev_polynomials(k-2, *args)

def chebychev_approx(degrees, coeff, *args):
    result = 0
    for i, sub_grid in enumerate(degrees):
        for j, grid_point in enumerate(sub_grid):
            sub_result = coeff[i][j]
            for k, arg in enumerate(*args):
                sub_result *= chebychev_polynomials(grid_point[k], arg)
            result += sub_result
    return result