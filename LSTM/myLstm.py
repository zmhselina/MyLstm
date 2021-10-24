import numpy as np

def sigmoid(x):
 return 1.0/(1.0+np.exp(-x))

def initialize(*args):
    np.random.seed(0)
    return np.random.randn(*args)*np.sqrt(0.1)

class LSTM:
    def __init__(self, vocab_size, embedding_dims, hidden_dims, out_vocab_size):

        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.hideen_dims = hidden_dims
        self.out_vocab_size = out_vocab_size
        self.concate_len = hidden_dims + embedding_dims

        self.Ww = initialize(self.embedding_dims, self.vocab_size)
        for gate in [self.Wg, self.Wi, self.Wf, self.Wo]:
            gate = initialize(self.hidden_dims, self.concate_len)
        self.Wv = initialize(self.out_vocab_size, self.hidden_dims)

        self.bw = initialize(self.embedding_dims)
        for gate in [self.bg, self.bi, self.bf, self.bo]:
            gate = initialize(self.hidden_dims)
        self.bv = initialize(self.out_vocab_size)

        self.deltaWw = np.zeros((self.embedding_dims, self.vocab_size))
        for gate in [self.deltaWg, self.deltaWi, self.deltaWf, self.deltaWo]:
            gate = np.zeros((self.hidden_dims, self.concate_len))
        self.deltaWv = np.zeros((self.out_vocab_size, self.hidden_dims))

        self.deltabw = np.zeros((self.embedding_dims))
        for gate in [self.deltabg, self.deltabi, self.deltabf, self.daltabo]:
            gate = np.zeros((self.hidden_dims))
        self.deltabv = np.zeros((self.out_vocab_size)

    def apply_delta(self, lr):

        self.Ww += lr*self.deltaWw
        self.Wg += lr*self.deltaWg
        self.Wi += lr*self.deltaWi
        self.Wf += lr*self.deltaWf
        self.Wo += lr*self.deltaWi
        self.Wv += lr*self.deltaWv

        self.deltaWw.fill(0.0)
        self.deltaWg.fill(0.0)
        self.deltaWi.fill(0.0)
        self.deltaWf.fill(0.0)
        self.deltaWo.fill(0.0)
        self.deltaWv.fill(0.0)

    def predict(self,x):

        g_gate = np.zeros((len(x) + 1, self.hidden_dims+self.embedding_dims))
        i_gate = np.zeros((len(x), self.hidden_dims+self.x_dim))
        f_gate = np.zeros((len(x), self.hidden_dims+self.x_dim))
        o_gate = np.zeros((len(x), self.hidden_dims+self.x_dim))
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
