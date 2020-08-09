import numpy as np

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)      * 2 - 1
        self.w = np.random.rand(n_conn, n_neur) * 2 - 1

class create_nn():
    def __init__(self, topology=[369, 512, 512, 128, 32, 9], model=None):
        self.sigm = (lambda x: 1 / (1 + np.e **(-x)),
                lambda x: x * (1 - x))

        self.relu = lambda x: np.maximun(0, x)

        self.l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
                        lambda Yp, Yr: (Yp - Yr))

        self.cost = 0
        self.out = []
        self.nn = [] # Estructura de datos

        if model==None:
            for l, layer in enumerate(topology[:-1]):
                self.nn.append(neural_layer(topology[l], topology[l+1], self.sigm))
        else:
            self.nn = model.nn[:] 

    def train(self, X, Y, lr=0.05, train=True):
        self.out = [(None, X)]
        # Froward pass
        for l, layer in enumerate(self.nn):
            z = np.matmul(self.out[-1][1], self.nn[l].w) + self.nn[l].b
            a = self.nn[l].act_f[0](z)
        
            self.out.append((z, a)) # Suma ponderada y activacion

        if train:
            # Backward pass
            self.deltas = []
            for l in reversed(range(0, len(self.nn))):
				z = self.out[l+1][0]
				a = self.out[l+1][1]
				if l == len(self.nn)-1:
					self.deltas.insert(0, self.l2_cost[1](a, Y) * self.nn[l].act_f[1](a))
				else:
					self.deltas.insert(0, np.matmul(self.deltas[0], _w.T) * self.nn[l].act_f[1](a))

				_w = self.nn[l].w

				# Gradient descent
				self.nn[l].b = self.nn[l].b - np.mean(self.deltas[0], axis=0, keepdims=True) * lr
				self.nn[l].w = self.nn[l].w - np.matmul(self.out[l][1].T, self.deltas[0]) * lr

            self.cost =  self.l2_cost[0](self.out[-1][1], Y)
        return self.out[-1][1]
