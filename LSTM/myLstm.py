import numpy as np

def sigmoid(x):
 return 1.0/(1.0+np.exp(-x))


class LSTM:

 def __init__(x_dim,hidden_dim):
  
  self.x_dim = x_dim
  self.hideen_dim = hidden_dim

  self.Wg = np.random.randn(self.hidden_dims, self.x_dim+self.hidden_dims) * np.sqrt(0.1)
  self.Wi = np.random.randn(self.hidden_dims, self.x_dim+self.hidden_dims) * np.sqrt(0.1)
  self.Wf = np.random.randn(self.hidden_dims, self.x_dim+self.hidden_dims) * np.sqrt(0.1)
  self.Wo = np.random.randn(self.hidden_dims, self.x_dim+self.hidden_dims) * np.sqrt(0.1)


  self.deltaWg = np.zeros((self.hidden_dims, self.x_dim+self.hidden_dims)
  self.deltaWi = np.zeros((self.hidden_dims, self.x_dim+self.hidden_dims)
  self.deltaWf = np.zeros((self.hidden_dims, self.x_dim+self.hidden_dims)
  self.deltaWo = np.zeros((self.hidden_dims, self.x_dim+self.hidden_dims)

  def apply_delta(self, lr):
  
  self.Wg += lr*self.deltaWg
  self.Wi += lr*self.deltaWi
  self.Wf += lr*self.deltaWf
  self.Wo += lr*self.deltaWi

  self.deltaWg.fill(0.0)
  self.deltaWi.fill(0.0)
  self.deltaWf.fill(0.0)
  self.deltaWo.fill(0.0)

 def predict(self,x):
  
  g_gate = np.zeros((len(x) + 1, self.hidden_dims+self.x_dim))
  i_gate = np.zeros((len(x) + 1, self.hidden_dims+self.x_dim))
  f_gate = np.zeros((len(x) + 1, self.hidden_dims+self.x_dim))
  o_gate = np.zeros((len(x) + 1, self.hidden_dims+self.x_dim))
  s = np.zeros((len(x) + 1, self.hidden_dims))
  ct = np.zeros((len(x) + 1, self.hidden_dims))
  y = np.zeros((len(x) + 1, self.hidden_dims))

   for t in range(len(x)):
    g_gate[t] = np.tanh(np.dot(self.Wg, s[t-1].T))
    i_gate[t] = sigmoid(np.dot(self.Wi, s[t-1].T))
    f_gate[t] = sigmoid(np.dot(self.Wf, s[t-1].T))
    o_gate[t] = sigmoid(np.dot(self.Wi, s[t-1].T))
    
    ct[t] = f_gate[

 def calculate_gradient(self,x
 

 def calculate_loss(self,

 
 def train(self,
