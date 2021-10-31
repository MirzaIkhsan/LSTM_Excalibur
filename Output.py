import numpy as np
from ActivationFunction import ActivationFunction

class Output():
  def __init__(self, U, W, bias=1):
    self.U = U
    self.W = W
    #self.ct = ct
    self.bias = bias
    
  def score_ot(self, x, h_prev):
    self.ot = ActivationFunction.sigmoid_num(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bias)
    return self.ot

  def score_ht(self, ct, x, h_prev):
    self.ht = np.array(np.dot(self.score_ot(x, h_prev), np.tanh(ct)))
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
