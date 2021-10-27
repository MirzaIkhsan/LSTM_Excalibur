import numpy as np
import math

class ActivationFunction:
    @staticmethod
    def linear(net):
        '''
        Menghitung fungsi aktivasi linear dengan menggunakan rumus
        dot product antara input neuron dengan masing-masing bobot
        '''
        return net

    @staticmethod
    def sigmoid(net):
        '''
        Menghitung fungsi aktivasi sigmoid dengan menggunakan rumus
        perhitungan logistik
        '''
        res = []
        for items in net:
            res.append(1 / (1 + np.exp(-items)))
        return res

    @staticmethod
    def relu(net):
        '''
        Menghitung fungsi aktivasi linear dengan menggunakan rumus
        dot product antara input neuron dengan masing-masing bobot
        Apabila menghasilkan nilai negatif maka akan dipilih nol
        '''
        return np.maximum(0, net)

    @staticmethod
    def softmax(net):
        '''
        Menghitung fungsi aktivasi softmax untuk setiap nilai
        pada net
        '''
        # net = np.array(net, dtype=np.float128)
        exp_net = [np.exp(net_el) for net_el in net]
        sum_of_exp_net = sum(exp_net)
        return [exp_net_el/sum_of_exp_net for exp_net_el in exp_net]
        # return 

    @staticmethod
    def get_by_name(func_name):
        funcs = {
            'linear': ActivationFunction.linear,
            'relu': ActivationFunction.relu,
            'sigmoid': ActivationFunction.sigmoid,
            'softmax': ActivationFunction.softmax,
        }
        return funcs[func_name] if func_name in funcs else None

        