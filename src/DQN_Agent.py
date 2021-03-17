from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import numpy as np

class DQN_Agent():
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.model = self.create_model()
        self.model.set_weights(load_model('/home/lucas/catkin_ws/src/proyecto_de_grado/src/DQL/stage_3_2980.h5').get_weights())
    
    def create_model(self):
        modelo = Sequential()
        dropout = 0.2

        modelo.add(Dense(64, input_shape=(28,), activation='relu', kernel_initializer='lecun_uniform'))

        modelo.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        modelo.add(Dropout(dropout))

        modelo.add(Dense(5, kernel_initializer='lecun_uniform'))
        modelo.add(Activation('linear'))
        modelo.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06))
        modelo.summary()

        return modelo
    
    def getAction(self, state):
        estado = np.array(state)
        self.q_value = self.model.predict(estado.reshape(1, 28))
        return np.argmax(self.q_value[0])