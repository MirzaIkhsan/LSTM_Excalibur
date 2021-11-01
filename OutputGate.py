import numpy as np
from ActivationFunction import ActivationFunction

class OutputGate():
  def __init__(self, U, W, bias):
    self.U = U
    self.W = W
    self.bias = np.array([[bias]])
    
  def score_ot(self, x, h_prev):
    self.ot = ActivationFunction.sigmoid_num(np.matmul(self.U, x.transpose()) + np.matmul(self.W, h_prev) + self.bias)

    return self.ot

  def score_ht(self, ct, x, h_prev):
    self.ht = np.dot(self.score_ot(x, h_prev), np.tanh(ct))
    return self.ht

if __name__ == "__main__":
  xt = np.array([[1, 2]])
  uo = np.array([[0.6, 0.4]])
  wo = np.array([[0.25]])
  h_prev = np.array([[0]])
  bo = 0.1
  ct = np.array([[0.7857261484]])

  output = OutputGate(uo, wo, bo)

  print("Output: ", output.score_ot(xt,h_prev))

  print("Hidden: ", output.score_ht(ct, xt, h_prev))
