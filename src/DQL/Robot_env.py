import numpy as np
import pygame
from Agent import Agent
from Spot import Spot
import time

class Robot_env():
    
    def __init__(self):
        """ Define the environment constant """ 
        self.WIDTH = 650
        self.HEIGHT = 500
         
        # Size of the environment
        self.COLS = 32
        self.ROWS = 32

        self.W = (self.WIDTH - 10) / self.COLS
        self.H = (self.HEIGHT - 10) / self.ROWS

        self.START_X = 27
        self.START_Y = 7
        self.END_X = 3
        self.END_Y = 28

        self.level = 1
        self.reward = -1
        self.MOVE_PENALTY = 1
        self.OBSTACLE_PENALTY = 300
        self.END_POINT_REWARD = 30
        self.OBSERVATION_SPACE_VALUES = (self.COLS, self.ROWS, 9)
        self.ACTION_SPACE_SIZE = 9
        self.done = False

        # Spots used to draw the env
        self.spots = [[Spot(i, j) for j in range(32)] for i in range(32)]

        # Read the walls list
        walls_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Wall.txt', 'r')
        self.WALLS = []
        self.WALLS_LIST = []
        tmp = walls_file.read().split('\n')
        walls_file.close()
        for pair in tmp:
            aux = pair.split(',')
            self.WALLS_LIST.append([int(aux[0]), int(aux[1])])
            self.WALLS.append(Spot(int(aux[0]), int(aux[1])))
            self.WALLS[-1].color = pygame.Color(173, 0, 75)

        # Read the obstacles list
        obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt', 'r')
        self.OBSTACLES_LIST_1 = []
        self.OBSTACLES_LIST_2 = []
        tmp = obstacle_file.read().split('\n')
        obstacle_file.close()

        for pair in tmp:
            aux = pair.split(',')
            self.OBSTACLES_LIST_1.append([int(aux[0]) - 3, int(aux[1])])
            self.OBSTACLES_LIST_2.append([int(aux[0]) + 8, int(aux[1])])
            
    def reset(self):
        self.agent = Agent(self.START_X + np.random.randint(-5, 5), self.START_Y)
        self.done = False
        self.obstacles = []
        self.obstacles_list = []
        rand1 = np.random.randint(-4, 3)
        rand2 = np.random.randint(-4, 3)
        for obs1, obs2 in zip(self.OBSTACLES_LIST_1, self.OBSTACLES_LIST_2):
            self.obstacles.append(Spot(obs1[0] + rand1, obs1[1] + rand1))
            self.obstacles_list.append([obs1[0] + rand1, obs1[1] + rand1])
            self.obstacles.append(Spot(obs2[0] + rand2, obs2[1] + rand2)) 
            self.obstacles_list.append([obs2[0] + rand2, obs2[1] + rand2])

        if self.level <= 5:
            self.end = Spot(self.agent.x, self.agent.y + self.level)
        else:
            self.end = Spot(15 + np.random.randint(-9, 9), 28)
        self.episode_step = 0
        self.observation = self.agent.view(self.obstacles_list, self.WALLS_LIST, self.end)

        return np.array(self.observation)

    def step(self, action):
        self.episode_step += 1
        self.agent.go_to(action)
        self.observation = self.agent.view(self.obstacles_list, self.WALLS_LIST, self.end)
        self.reward = -self.MOVE_PENALTY
        
        if [self.agent.x, self.agent.y] in self.WALLS_LIST or [self.agent.x, self.agent.y] in self.obstacles_list:
            self.reward = -self.OBSTACLE_PENALTY
            self.done = True
            self.level = 1

        elif [self.agent.x, self.agent.y] == [self.end.x, self.end.y]:
            self.reward += self.END_POINT_REWARD
            self.level += 1
            self.reset()

        
        elif self.agent.x > self.ROWS or self.agent.x < 0 or self.agent.y > self.COLS or self.agent.y < 0:
            self.done = True
            self.reward = self.OBSTACLE_PENALTY
            self.level = 1

        return (self.observation, self.reward, self.done)

    def render(self):
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('DQL_Env')
        pygame.draw.rect(self.screen, (148, 148, 148), (0, 0, self.WIDTH, self.HEIGHT))

        #Draw the grid
        for i in range(self.COLS):
            for j in range(self.ROWS):
                pygame.draw.rect(self.screen, (0, 0, 255), (self.spots[i][j].x * self.W, 1 + self.spots[i][j].y * self.H, self.W, self.H), 1)

        # Draw the agent
        pygame.draw.rect(self.screen, self.agent.color, (self.agent.x * self.W, 1 + self.agent.y * self.H, self.W, self.H))

        # Draw the end spot
        pygame.draw.rect(self.screen, (255, 234, 0), (self.end.x * self.W, 1 + self.end.y * self.H, self.W, self.H))

        # Draw what the robot sees
        for x, y in self.agent.obs_to_draw:
            pygame.draw.rect(self.screen, (255, 255, 255), (x * self.W, 1 + y * self.H, self.W, self.H))

        pygame.display.update()
