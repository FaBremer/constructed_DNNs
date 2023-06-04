import numpy as np
from . import dnn_class as nn

def relu(x):
    # x: input - may be list, np.ndarray, int or float
    # returns ReLU(x) = max(0,x) (or componentwise if x is list or np.ndarray)
    if not isinstance(x, (np.ndarray, list)):
        x = [x]
    y = np.maximum(x, np.zeros(len(x)))
    if len(y) == 1:
        y = y[0]
    return y

def identity_dnn(d):
    # d: dimension
    # returns DNN that emulates identity map in dimension d
    a = np.identity(d)
    b = -a
    id_dnn = nn.NeuralNetwork(d, [2*d], d)
    id_dnn.weights[0][:d, :] = a
    id_dnn.weights[0][d:, :] = b
    id_dnn.weights[1][:, :d] = a
    id_dnn.weights[1][:, d:] = b
    return id_dnn

def identity_dnn_n(d,n):
    # d: dimension
    # n: depth
    # returns DNN that emulates identity map in dimension d with depth n
    res = identity_dnn(d)
    base = identity_dnn(d)
    for _ in range(n-1):
        res = res.concatenate(base)
    return res

def pli(x, m):
    # not strictly necessary for the construction, but usable for testing
    # m: number of subdivisions
    # x: point to evaluate
    # returns piecewise linear interpolant for x**2
    points = np.arange(1+2*m)
    values = []
    for point in points:
        values.append(point**2)
    return np.interp(x, points, values)

def two_numbers(x,y,delta=.001):
    # multiplies two numbers with error \leq delta in a way like the DNN should
    # not strictly necessary for the construction, but usable for testing
    # x,y: numbers to multiply
    # delta: error bound
    # returns x*y with error \leq delta
    m = np.ceil(-np.log2(delta*.5))
    return .5*(16*pli(np.absolute(x+y)*.25,m) - 4*pli(np.absolute(x)*.5,m) - 4*pli(np.absolute(y)*.5,m))

def g_dnn():
    # Sawtooth function as DNN
    # returns DNN that emulates sawtooth function
    g = nn.NeuralNetwork(1, [3], 1)
    g.weights[0] =np.array([[2],[4],[2]])
    g.biases[0] = np.array([0, -2, -2])
    g.weights[1] = np.array([[1,-1,1]])
    return g

def g_m_dnn(m):
    # m : number of concatenations of g
    # returns m concatenations of g as DNN
    g_m = g_dnn()
    for i in (range(m-1)):
        A = g_dnn()
        g_m = g_m.concatenate(A)
    return g_m

def f_m(m,x):
    # not strictly necessary for the construction, but usable for testing
    # m: number of subdivisions
    # x: point to evaluate
    # # returns f_m, i.e. the squaring function, but not as DNN
    if m == 0:
        return relu(x)
    else:
        return (f_m(m-1,x) - (g_m_dnn(m).realize(x)/2**(2*m)))

def f_m_dnn(m):
    # m : number of subdivisions
    # returns f_m, i.e. the squaring function, as DNN
    if m == 0:
        #if m=0 return relu by definition
        f_1 = nn.NeuralNetwork(1,[1],1)
        f_1.weights[0] = [1]
        f_1.weights[1] = [1]
        return f_1
    base = g_m_dnn(m)
    base.weights[0] = np.append(base.weights[0],[[1]], axis=0)
    base.biases[0] = np.append(base.biases[0],0)
    base.hidden_dims[0] = base.hidden_dims[0] +1
    for i in range(1,m):
        base.weights[i] = np.c_[base.weights[i], np.zeros(3)]
        base.weights[i] = np.r_[base.weights[i], [[-2**(-2*i), 2**(-2*i), -2**(-2*i), 1]]]
        base.biases[i] = np.append(base.biases[i],0)
        base.hidden_dims[i] = base.hidden_dims[i] +1
    base.weights[m] = np.array([[-2**(-2*m), 2**(-2*m), -2**(-2*m), 1]])
    return base

def abs_dnn():
    # returns absolute function as DNN
    abs = nn.NeuralNetwork(1,[2], 1)
    abs.weights[0] = np.array([[1],[-1]])
    abs.weights[1] = np.array([[1,1]])
    return abs

def abs_sum_dnn():
    # returns absolute value of the sum of two inputs as DNN
    abs2 = nn.NeuralNetwork(2, [2], 1)
    abs2.weights[0] = np.array([[1,1],[-1,-1]])
    abs2.weights[1] = np.array([[1,1]])
    return abs2

def prepare_input_dnn(M=1):
    # prepaes two numbers a and b for multiplication by computing absolutes and absolute of sum
    # M: a,b \in [-M,M]
    # returns DNN that prepares input for multiplication
    res = nn.NeuralNetwork(2, [6], 3)
    res.weights[0] = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1]])
    res.weights[1] = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])*(.5*(M**(-1))) #.5*M**(-1) factor as described in proof of Lemma 2.5
    return res

def prepare_ouput_dnn(M=1):
    # prepares output of two_mult_dnn
    # M: a,b \in [-M,M]
    # returns DNN that prepares output of multiplication
    res = nn.NeuralNetwork(3,[],1)
    res.weights[0] = np.array([[-2,-2,2]])*(M**2) # factor 2M**2 as described in proof of Lemma 2.5
    return res

def two_mult_dnn(delta=.001, M=1):
    # delta: error bound
    # M: x,y \in [-M,M]
    # returns DNN that multiplies two numbers a,b \in [-M,M] s.th. max error is \leq delta
    m = int(np.ceil(-np.log2(delta*.5*(M**(-1)))))
    temp1 = prepare_input_dnn(M)
    temp2 = f_m_dnn(m)
    temp3 = temp2.n_parallelize([f_m_dnn(m),f_m_dnn(m)])
    temp4 = temp3.concatenate(temp1)
    temp5 = prepare_ouput_dnn(M)
    res = temp5.concatenate(temp4)
    return res

def r_dnn(l, n_tilde, n, delta):
    # l: number of concatenations
    # n_tilde: length of input (needed, aka a power of 2)
    # n: length of input (actual)
    # delta: error bound
    # returns DNN that multiplies n numbers with error \leq delta
    if n_tilde == n:
        if l == 1:
            temp = []
            if n_tilde == 2:
                return two_mult_dnn(delta*(n_tilde**(-2)), 2)
            base = two_mult_dnn(delta*(n_tilde**(-2)), 2)
            for _ in range(int(n_tilde/2)-1):
                temp.append(two_mult_dnn(delta*(n_tilde**(-2)), 2))
            res = base.n_parallelize(temp)
            return res
        else:
            return r_dnn(l-1, int(.5*n_tilde), n, delta).concatenate(r_dnn(1, n_tilde, n, delta))
    # If n is not a power of 2, we need to add ones through the biases to make it work
    else:
        res = r_dnn(l, n_tilde, n_tilde, delta)
        for i in range(n_tilde - n):
            res.weights[0] = res.weights[0][:,:n]
            if n%2 != 0:
                res.biases[0][3*n-1:3*(n+1)] = [1,-1,1,-1]
                for i in range(n, n_tilde-2,2):
                    res.biases[0][3*(i+1):3*(i+3)] = [1,-1,1,-1,2,-2]
            else:
                for i in range(n, n_tilde,2):
                    res.biases[0][3*i:3*(i+2)] = [1,-1,1,-1,2,-2]
        return res

def mult_dnn(n, delta=.001, M=1):
    # n: number of numbers to multiply
    # delta: error bound
    # M: numbers to be multiplied \in [-M,M]
    # returns DNN that multiplies n numbers with error \leq delta
    if n ==1:
        return identity_dnn(1)  
    if M == 1:
        k = 1
        state = True
        while state:
            if 2**k >= n:
                n_tilde = 2**k
                state = False
            else:
                k +=1
        return r_dnn(int(np.log2(n_tilde)), n_tilde, n, delta)
    else:
        result = mult_dnn(n, delta*(M**(-n)), 1)
        result.weights[0] = result.weights[0]*(M**(-1))
        result.weights[-1] = result.weights[-1]*(M**n)
        result.biases[-1] = result.biases[-1]*(M**n)
        return result

#only for checking if M>1 works in mult_dnn, won't be used in further code
def realize_mult_dnn(x, delta =.001, M=1):
    # x: list of numbers to multiply
    # delta: error bound
    # M: a \in [-M,M] for a \in x
    # returns realization of mult_dnn with M\geq1 for x
    if isinstance(x, np.float64):
        x = [x]
    elif not isinstance(x, list):
        x = x.tolist()
    n = len(x)
    x = np.array(x)
    if M > 1:
        return realize_mult_dnn(M**(-1)*x, delta*(M**(-n)), 1)*(M**n)
    else:
        if n == 1:
            return x[0]
        else:
            return realize_mult_dnn(mult_dnn(n, delta).realize(x), delta, 1)