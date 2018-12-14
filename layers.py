import numpy as np
import math

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.maximum(input, 0.0)

    def backward(self, grad_output):
        '''Your codes here'''
        gradient = grad_output
        gradient[self._saved_tensor < 0.0] = 0.0
        return gradient

# implementation of ELU


class ELU(Layer):
    def __init__(self, name, ai):
        super(ELU, self).__init__(name)
        self.ai = ai
        self.judge = None

    def forward(self, input):
        self._saved_for_backward(input)
        output = input
        h1 = input.shape[0]
        h2 = input.shape[1]
        for i in range(h1):
            for j in range(h2):
                if (input[i][j] < 0):
                    output[i][j] = self.ai * (math.exp(input[i][j])-1)

        return output

    def backward(self, grad_output):
        gradient = grad_output
        h1 = grad_output.shape[0]
        h2 = grad_output.shape[1]
        for i in range(h1):
            for j in range(h2):
                if (self._saved_tensor[i][j] < 0.0):
                    gradient[i][j] *= self.ai * math.exp(self._saved_tensor[i][j])

        return gradient


# leaky_relu: ai is the adjustable args<1

class Leaky_Relu(Layer):
    def __init__(self, name, ai):
        super(Leaky_Relu, self).__init__(name)
        self.ai = ai

    def forward(self, input):
        self._saved_for_backward(input)
        output = input
        output[input < 0.0] /= self.ai
        return output

    def backward(self, grad_output):
        gradient = grad_output
        gradient[self._saved_tensor < 0.0] /= self.ai
        return gradient


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        fx = 1.0 / (1.0 + np.exp(-input))
        self._saved_for_backward(fx)  # saved f(x) not x, better for bp
        return fx  # sigmoid function

    def backward(self, grad_output):
        '''Your codes here'''
        gradient = grad_output * self._saved_tensor * (1 - self._saved_tensor)
        return gradient


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.matmul(input, self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''  # grad_output shape (100,10)
        self.grad_W = np.matmul(self._saved_tensor.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        return np.matmul(grad_output, self.W.T)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
