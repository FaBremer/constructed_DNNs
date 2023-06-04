import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim):
        # input_dim: dimension of input
        # hidden_dims: list of dimensions (number of neurons) of hidden layers
        # output_dim: dimension of output
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.weights.append(np.zeros((dims[i+1], dims[i])))
            self.biases.append(np.zeros((dims[i+1])))

    def realize(self, x):
        # x: input (may be a vector or scalar)
        # returns realization of DNN at x with ReLU activation function
        if (not isinstance(x, np.ndarray)) and (not isinstance(x, list)):
            a = [x]
        else:
            a = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.matmul(w, a) + b
            zero = np.zeros(len(z))
            a = np.maximum(zero, z) #ReLU
        y = np.matmul(self.weights[-1], a) + self.biases[-1]
        if len(y) == 1:
            y = y[0]
        return y
    
    def count_depth_size(self):
        # returns depth (number of hidden layers) and size (number of nonzeros in weights and biases)
        depth = len(self.weights)-1
        size = 0
        for i in range(depth):
            size += np.count_nonzero(self.weights[i])
            size += np.count_nonzero(self.biases[i])
        return depth, size

    def combine(self, nn1, nn2):
        # Combine weight and bias matrices from two networks for concatenation
        # nn1: first network
        # nn2: second network
        # returns combined weight and bias matrices
        if nn1.weights[0].shape[1] != nn2.weights[-1].shape[0]:
            raise ValueError(f"The dimensions of the weight matrices {nn1.weights[0].shape[1]} and {nn2.weights[-1].shape[0]} are not compatible")
        
        combined_weights = np.matmul(nn1.weights[0], nn2.weights[-1])
        combined_biases = np.matmul(nn1.weights[0], nn2.biases[-1]) + nn1.biases[0]
        
        return combined_weights, combined_biases


    def concatenate(self, other):
        # Concatenate two DNNs, s.th. self(other(x))
        # other: other DNN
        # returns concatenated DNN
        if self.input_dim != other.output_dim:
            raise ValueError(f"The input dimension {self.input_dim} of the first network must match the output dimension {other.output_dim} of the second network")
        middle_layer_weights, middle_layer_biases = self.combine(self, other)
        input_dim = other.input_dim
        hidden_dims = other.hidden_dims + self.hidden_dims
        output_dim = self.output_dim
        concatenated_nn = NeuralNetwork(input_dim, hidden_dims, output_dim)

        # Copy weights and biases from self and other networks
        for i in range(len(other.weights)-1):
            concatenated_nn.weights[i] = other.weights[i]
            concatenated_nn.biases[i] = other.biases[i]
        concatenated_nn.weights[len(other.weights)-1] = middle_layer_weights
        concatenated_nn.biases[len(other.weights)-1] = middle_layer_biases
        for i in range(len(self.weights)-1):
            concatenated_nn.weights[i+len(other.weights)]= self.weights[i+1]
            concatenated_nn.biases[i+len(other.biases)] = self.biases[i+1]
        
        return concatenated_nn
    
    def parallelize(self, other):
        # Parallelize two DNNs
        # other: other DNN
        # returns parallelized DNN
        if len(self.hidden_dims) != len(other.hidden_dims):
            raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {other.hidden_dims} of the second network")
        input_dim = self.input_dim + other.input_dim
        hidden_dims = [sum(x) for x in zip(self.hidden_dims,  other.hidden_dims)]
        output_dim = self.output_dim + other.output_dim
        parallel_nn = NeuralNetwork(input_dim, hidden_dims, output_dim)
        
        # Copy weights and biases from self and other networks
        for i in range(len(self.weights)):
            parallel_nn.weights[i][:self.weights[i].shape[0], :self.weights[i].shape[1]] = self.weights[i]
            parallel_nn.weights[i][self.weights[i].shape[0]:,self.weights[i].shape[1]:] = other.weights[i]
            parallel_nn.biases[i][:len(self.biases[i])] = self.biases[i]
            parallel_nn.biases[i][len(self.biases[i]):] = other.biases[i]
        return parallel_nn


    def n_parallelize(self, others):
        # Parallelize n DNNs
        # others: list with all networks
        # returns parallelized DNN
        if len(others) == 0:
            return self
        if len(self.hidden_dims) != len(others[0].hidden_dims):
            raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {others[0].hidden_dims} of the second network")
        
        input_dim = self.input_dim
        hidden_dims = self.hidden_dims
        output_dim = self.output_dim
        
        for other in others:
            if len(self.hidden_dims) != len(other.hidden_dims):
                raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {other.hidden_dims} of the second network")
            
            input_dim += other.input_dim
            hidden_dims = [sum(x) for x in zip(hidden_dims, other.hidden_dims)]
            output_dim += other.output_dim
        
        parallel_nn = NeuralNetwork(input_dim, hidden_dims, output_dim)
        
        # Copy weights and biases from self and other networks
        for i in range(len(self.weights)):
            parallel_nn.weights[i][:self.weights[i].shape[0], :self.weights[i].shape[1]] = self.weights[i]
            parallel_nn.biases[i][:len(self.biases[i])] = self.biases[i]
            
            start_row = self.weights[i].shape[0]
            start_col = self.weights[i].shape[1]
            
            for other in others:
                parallel_nn.weights[i][start_row:start_row+other.weights[i].shape[0], start_col:start_col+other.weights[i].shape[1]] = other.weights[i]
                parallel_nn.biases[i][start_row:start_row+other.weights[i].shape[0]] = other.biases[i]
                
                start_row += other.weights[i].shape[0]
                start_col += other.weights[i].shape[1]
        
        return parallel_nn

    """
        # This version of n_parallelize is recursive and therefore abandoned
        def n_parallelize(self, others):
            # Parallelize n DNNs 
            # others should be list with all networks
            if len(others) == 1:
                return self.parallelize(others[0])
            else:
                res = self.parallelize(others[0])
                others.pop(0)
                return res.n_parallelize(others)
    """     

    def undistinct_parallelize(self, other):
        # Parallelize two DNNs with undistinct inputs
        # other: other DNN
        # returns undistinctly parallelized DNN
        if len(self.hidden_dims) != len(other.hidden_dims):
            raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {other.hidden_dims} of the second network")
        if self.input_dim != other.input_dim:
            raise ValueError(f"The input dimension {self.input_dim} of the first network must match the input dimension {other.input_dim} of the second network")
        input_dim = self.input_dim
        hidden_dims = [sum(x) for x in zip(self.hidden_dims,  other.hidden_dims)]
        output_dim = self.output_dim + other.output_dim
        parallel_nn = NeuralNetwork(input_dim, hidden_dims, output_dim)
        
        # Copy weights and biases from self and other networks
        parallel_nn.weights[0][:self.weights[0].shape[0], :] = self.weights[0]
        parallel_nn.weights[0][self.weights[0].shape[0]:, :] = other.weights[0]
        parallel_nn.biases[0][:len(self.biases[0])] = self.biases[0]
        parallel_nn.biases[0][len(self.biases[0]):] = other.biases[0]
        for i in range(1,len(self.weights)):
            parallel_nn.weights[i][:self.weights[i].shape[0], :self.weights[i].shape[1]] = self.weights[i]
            parallel_nn.weights[i][self.weights[i].shape[0]:,self.weights[i].shape[1]:] = other.weights[i]
            parallel_nn.biases[i][:len(self.biases[i])] = self.biases[i]
            parallel_nn.biases[i][len(self.biases[i]):] = other.biases[i]
        return parallel_nn

    def n_undistinct_parallelize(self, others):
        # Parallelize n DNNs with undistinct inputs
        # others: list with all networks
        # returns undistinctly parallelized DNN
        if len(others) == 0:
            return self
        if len(self.hidden_dims) != len(others[0].hidden_dims):
            raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {others[0].hidden_dims} of the second network")
        if self.input_dim != others[0].input_dim:
            raise ValueError(f"The input dimension {self.input_dim} of the first network must match the input dimension {others[0].input_dim} of the second network")

        input_dim = self.input_dim
        hidden_dims = self.hidden_dims
        output_dim = self.output_dim
        
        for other in others:
            if len(self.hidden_dims) != len(other.hidden_dims):
                raise ValueError(f"The depth {self.hidden_dims} of the first network must match the depth {other.hidden_dims} of the second network")
            if self.input_dim != other.input_dim:
                raise ValueError(f"The input dimension {self.input_dim} of the first network must match the input dimension {other.input_dim} of the second network")
            
            hidden_dims = [sum(x) for x in zip(hidden_dims, other.hidden_dims)]
            output_dim += other.output_dim
        
        parallel_nn = NeuralNetwork(input_dim, hidden_dims, output_dim)
        parallel_nn.weights[0][:self.weights[0].shape[0], :] = self.weights[0]
        parallel_nn.biases[0][:len(self.biases[0])] = self.biases[0]

        start_row = 0
        
        # Copy weights and biases from self and other networks
        for i in range(1,len(self.weights)):
            parallel_nn.weights[i][:self.weights[i].shape[0], :self.weights[i].shape[1]] = self.weights[i]
            parallel_nn.biases[i][:len(self.biases[i])] = self.biases[i]
            
            start_row = self.weights[i].shape[0]
            start_col = self.weights[i].shape[1]
            
            for other in others:
                if i == 1:
                    parallel_nn.weights[0][start_row:start_row+other.weights[0].shape[0],:] = other.weights[0]
                    parallel_nn.biases[0][start_row:start_row+other.weights[0].shape[0]] = other.biases[0]

                parallel_nn.weights[i][start_row:start_row+other.weights[i].shape[0], start_col:start_col+other.weights[i].shape[1]] = other.weights[i]
                parallel_nn.biases[i][start_row:start_row+other.weights[i].shape[0]] = other.biases[i]
                
                start_row += other.weights[i].shape[0]
                start_col += other.weights[i].shape[1]
        
        return parallel_nn


"""
    # This version of n_parallelize is recursive and therefore abandoned
    def n_undistinct_parallelize(self, others):
            # Parallelize n DNNs with undistinct inputs 
            # others should be list with all networks
            if len(others) == 1:
                return self.undistinct_parallelize(others[0])
            else:
                res = self.undistinct_parallelize(others[0])
                others.pop(0)
                return res.n_undistinct_parallelize(others)
"""

