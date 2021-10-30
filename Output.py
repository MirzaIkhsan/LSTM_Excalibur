import numpy as np
from ActivationFunction import ActivationFunction

class Output():
  def __init__(self, U, W, bias=1):
    self.U = U
    self.W = W
    #self.ct = ct
    self.bias = bias
    
  def ot(self, x, h_prev):
    self.ot = ActivationFunction.sigmoid(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bias)
    return self.ot

  def ht(self, ct):
    self.ht = np.array([np.dot(self.ot, np.tanh(self.ct))])
    return self.ht

if __name__ == "__main__":
  xt = np.array([1, 2])
  uo = np.array([0.6, 0.4])
  wo = np.array([0.25])
  h_prev = np.array([0])
  bo = np.array([0.1])
  ct = np.array([0.7857261484])

  output = Output(uo, wo, ct, bo)

  print("Output: ", output.ot(xt,h_prev))

  print("Hidden: ", output.ht())
