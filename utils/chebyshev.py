import numpy as np
from .helper_functions import generate_grid_points

class Chebyshev:

    def __init__(self, fun, degree: int, dim: int):
        # fun: function to approximate
        # degree: degree of approximation
        # dim: dimension
        self.fun = fun
        self.degree = degree
        self.dim = dim
        self.grid = self._grid()

    def _grid(self,degree = None):
        # degree: degree for grid, if None, use self.degree
        # returns grid of points
        if degree is None:
            degree = self.degree
        else:
            degree = degree
        return np.asarray(generate_grid_points(degree, self.dim))

    def get_degrees(self):
        # Used for reshaping the grid (not necessary for the construction)
        return self.grid.reshape(self.degree + 1, self.degree + 1, self.dim)
    
    def _cos_grid(self, grid = None, degree = None):
        # grid: grid to transform, if None, use self.grid
        # degree: degree for transformation, if None, use self.degree
        # returns grid transformed to cosines as in Chapter 2
        if grid is None:
            grid = self.grid
        if degree is None:
            degree = self.degree
        points = np.array(grid)
        x = points[:, 0]
        z = np.cos(x*np.pi/degree) # first dimension
        for i in range(1, points.shape[1]):
            x = points[:, i]
            z = np.append(z,np.cos(x*np.pi/degree)) # other dimensions
        return z.reshape(points.shape[1],points.shape[0]).T
        
    def get_transformed_grid(self):
        # returns grid transformed to cosines as in Chapter 2
        return self._cos_grid()

    def _indicator(self):
        # returns indicator function for the grid as in Chapter 2
        kk = [l for l in self.grid]
        out = []
        for l in kk:
            S = sum(1 for i in l if i != 0 and i != self.degree)
            out.append(S)
        return out

    def polynomials_old(self, k, *args):
        # Same as below, but recursive, therefore not used
        if k == 0:
            return 1
        elif k == 1:
            return np.prod(args)
        else:
            return 2*np.prod(args)*self.polynomials(k-1, *args) - self.polynomials(k-2, *args)
    
    def polynomials(self, k, x):
        # k: degree of polynomial
        # x: point to evaluate at
        #returns Chebyshev polynomials of the first kind evaluated at x
        return np.cos(k * np.arccos(x))

    def _interpolate(self, coeff, x):
        # Interpolates the function at the given point
        # coeff: coefficients of the interpolation
        # x: point to evaluate at (single point)
        # returns the interpolation at x
        result = 0
        l, k = self.grid.shape # k is dimension, l-1 is degree of approximation    
        if k != len(x):
            raise Exception(f"Lenght of input must match Dimenssion {self.dim}!")
        for i in range(l):
            sub_result = coeff[i]
            grid_point = self.grid[i]
            for j, arg in enumerate(x):
                sub_result *= self.polynomials(grid_point[j], arg)
            result += sub_result
        return result
    
    def interpolate(self, x, coeff = None):
        # Interpolates the function at multiple given points (with shape (len(x),dim))
        # x: points to evaluate at
        # coeff: coefficients of the interpolation, if None, use self.coefficients()
        # returns the interpolations at x
        if coeff is None:
            grid, coeff = self.coefficients()
        k = self.dim
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0],k)
        if k != x.shape[1]:
            raise Exception(f"Shape of input must be (x,{self.dim}) for x > 1!")
        return np.asarray([self._interpolate(coeff, y) for y in x])
    
    def _cos_kj(self, grid_k, grid_j):
        # grid_k = self.grid
        # grid_j = self._grid(2*self.dim-1)
        # returns product of cosines as in Chapter 2
        kk = [kk for kk in grid_k]
        res = []
        for k in kk:
            inner = self._cos_grid(grid_j*k)
            inner = np.prod(inner,axis=1)
            res.append(inner)
        return np.array(res)
    
    def _sum_mult(self, f_hat, grid_cos):
        # f_hat: function to evaluate
        # grid_cos: grid to evaluate at (transformed to cosines)
        # returns sum of function evaluations and cosine products as in Chapter 2
        out = []
        for kj in grid_cos:
            prd = f_hat*kj
            out.append(np.sum(prd,axis=0))            
        return np.asarray(out)
    
    def coefficients(self, reshape = False):
        # reshape: if True, reshape the coefficients to the shape of the grid
        # returns grid and coefficients of the interpolation as in Chapter 2
        fac = self._indicator()
        fac = np.array([2**i for i in fac])
        fac = fac*(2*self.degree)**(-self.dim) #This is 2^{\mathbbm{1}_{S(n)_(k)}} * (2n)^{-k}
        grid_j = self._grid(2*self.degree-1)
        grid_cos_j = self._cos_grid(grid_j) 
        grid_cos_kj = self._cos_kj(self.grid,grid_j) 
        f_hat = self.fun(grid_cos_j) # Transpose for correct input
        if self.dim == 1:
            f_hat = f_hat.reshape((2*self.degree,))
        f_hat = self._sum_mult(f_hat,grid_cos_kj)
        ret = fac*f_hat # elemt-wise multiplication
        if reshape:
            self.get_degrees(), ret.reshape([self.degree +1 for i in range(self.dim)])
        return self.grid, ret

"""
Following: Not used in further code, but usefull for arbitrary functions
"""

def chebyshev_transform(x, a = -1.0, b = 1.0):
    return a + 0.5 * (b - a) * (x + 1.0)

def chebyshev_inv_transform(y, a = -1.0, b = 1.0):
    return 2 / (b-a) * (y-a) - 1.0