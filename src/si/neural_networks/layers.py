from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
         return self._input_shape

    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):

     
    