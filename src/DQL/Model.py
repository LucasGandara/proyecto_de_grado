import numpy as np
from Utils import initialize_parameters, LinearLayer, SigmoidLayer, compute_cost, set_parameters

class Sequential(object):
    def __init__(self, input_shape, n_out, ini_type='xavier', from_model=False, model=None):

        self.cost = 0

        # Layer 1
        self.Z1 = LinearLayer(input_shape=input_shape, n_out=512, ini_type='xavier')
        self.A1 = SigmoidLayer(self.Z1.Z.shape)

        # Layer 2
        self.Z2 = LinearLayer(input_shape=self.A1.A.shape, n_out=256,  ini_type='xavier')
        self.A2 = SigmoidLayer(self.Z2.Z.shape)

        # Layer 3
        self.Z3 = LinearLayer(input_shape=self.A2.A.shape, n_out=n_out, ini_type='xavier')
        self.A3 = SigmoidLayer(self.Z3.Z.shape)

        if from_model:
            self.Z1 = LinearLayer(input_shape=input_shape, n_out=512, ini_type='xavier', from_params=True, params=model.Z1.params)
            self.Z2 = LinearLayer(input_shape=self.A1.A.shape, n_out=256,  ini_type='xavier', from_params=True, params=model.Z2.params)
            self.Z3 = LinearLayer(input_shape=self.A2.A.shape, n_out=n_out, ini_type='xavier', from_params=True, params=model.Z3.params)

    def forward_prop(self, X):
        # --- Forward prop ---
        self.Z1.forward(X)
        self.A1.forward(self.Z1.Z)

        self.Z2.forward(self.A1.A)
        self.A2.forward(self.Z2.Z)

        self.Z3.forward(self.A2.A)
        self.A3.forward(self.Z3.Z)

        return self.A3.A

    def train(self, X_train, Y_train, LR, N_Epochs):
        for epoch in range(N_Epochs):
            for XX, YY in zip(X_train, Y_train):

                X = np.array([XX]).T    
                Y = np.array(YY)

                # Forward prop
                self.forward_prop(X)  

                # Compute Cost
                self.cost, dA3= compute_cost(Y=Y, Y_hat=self.A3.A)

                # ---back-prop
                self.A3.backward(dA3)
                self.Z3.backward(self.A3.dZ)

                self.A2.backward(self.Z3.dA_prev)
                self.Z2.backward(self.A2.dZ)

                self.A1.backward(self.Z2.dA_prev)
                self.Z1.backward(self.A1.dZ)

                # --- Update weights and bias
                self.Z3.update_params(learning_rate=LR)
                self.Z2.update_params(learning_rate=LR)
                self.Z1.update_params(learning_rate=LR)
        
    def predict(self, X):
        """
        helper function to predict on data using a neural net model layers
        Args:
            X: Data in shape (features x num_of_examples)
            Y: labels in shape ( label x num_of_examples)
            Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
            As: All Activation layers in form of a list e.g [A1,A2,...,An]
        Returns::
            p: predicted labels
            probas : raw probabilities
            accuracy: the number of correct predictions from total predictions
        """
        Zs = [self.Z1, self.Z2, self.Z3]
        As = [self.A1, self.A2, self.A3]

        m = X.shape[1]
        n = len(Zs)  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        Zs[0].forward(X)
        As[0].forward(Zs[0].Z)
        for i in range(1, n):
            Zs[i].forward(As[i-1].A)
            As[i].forward(Zs[i].Z)
        probas = As[n-1].A

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:  # 0.5 is threshold
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))

        return p