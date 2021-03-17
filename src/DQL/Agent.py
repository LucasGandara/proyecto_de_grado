import numpy as np
import random
import pygame
from Spot import Spot

class Agent(Spot):
    def __init__(self, x, y):
       Spot.__init__(self, x, y)
       self.range_of_view = 10
       self.obs = []
       self.obs_list = []
       self.obs_to_draw = []
       self.discrete_view = []
       self.color = pygame.Color(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
       self.distToFinalPoint = 0
       self.closestPath = 0
       self.lifepoints =  65
       self.directional_view = [0, 0, 0, 0, 0, 0, 0, 0] # This is the indicator of the direccion of the goal point [up, up-rigth, rigth, down-rigth, down, down-left, left, up-left]
       self.OBSERVATION_SIZE = 369
    
    # TODO: get some better names
    def move_to(self, x, y):
        """ Moves the agent to a given (x, y) 
            position """
        self.x = x
        self.y = y
    def go_to(self, n):
        if n == 0: # Stay still
            pass
        elif n == 1: # Go to the rigth
            self.x = self.x + 1
        elif n == 2: # Go to the top rigth
            self.x = self.x + 1
            self.y = self.y - 1
        elif n == 3: # Go up
            self.y = self.y - 1
        elif n == 4: # Go top left
            self.x = self.x - 1
            self.y = self.y - 1
        elif n == 5: # Go to the left
            self.x = self.x - 1
        elif n == 6: # Go down left
            self.y = self.y - 1
            self.x = self.x - 1
        elif n == 7: # Go down
            self.y = self.y + 1
        elif n == 8: # Go down ritgth
            self.x = self.x + 1
            self.y = self.y + 1
            
    def view(self, obss_list, walls_list, end):
        spots = [[Spot(i, j) for j in range(32)] for i in range(32)]
        for i in range(len(spots)):
            for j in range(len(spots[i])):
                spots[i][j].addNeighbors(spots, 1)
        """ This function allows the virtual agent to watch it's surroundings"""
        self.obs         = []
        self.obs_list    = []
        self.obs_to_draw = []
        self.discrete_view = [-1 for i in range(361)]
        self.directional_view = [0, 0, 0, 0, 0, 0, 0, 0] 

        for i in range(0, self.range_of_view):
            for j in range(-i, i + 1):
                for k in range(-i, i + 1):
                    try:
                        if [self.x + j, self.y + k] not in self.obs_list: # Append only once
                            self.obs.append(spots[self.x + j][self.y + k])
                            #obs_list menas view list
                            self.obs_list.append([self.x + j, self.y + k])
                    except IndexError:
                        pass

        for x, obstacle in enumerate(self.obs_list):
            if obstacle in walls_list or obstacle in obss_list:
                self.obs_to_draw.append(obstacle)
                self.discrete_view[x] = 1

        directions = [ [self.x, self.y-1]  , [self.x+1, self.y-1], [self.x+1, self.y],
                       [self.x+1, self.y+1], [self.x, self.y+1]  , [self.x-1, self.y+1],
                       [self.x-1, self.y]  , [self.x-1, self.y-1] ]
        distances = [] 
        for direction in directions:
            distances.append(np.linalg.norm(np.array(direction) - np.array([end.x, end.y])))                        
        # Set the minimun to 1
        self.directional_view[distances.index(min(distances))] = 1

        for direction in self.directional_view:
            pass
            # Uncomment this to add directional view
            #self.discrete_view.append(direction)

        return np.array(self.discrete_view)

        """ Draw the direction of the goal point

        for indicator, direction in zip(self.directional_view, directions):
            print 'indicator: ', indicator
            print 'direction: ', direction
            xx, yy = direction
            if indicator == 1:
                color = pygame.Color(0, 0, 0)
            elif indicator == 0:
                color = pygame.Color(255, 255, 255)

            pygame.draw.rect(screen, color, (xx * w, 1 + yy * h, w, h), 1)        
            pygame.display.update()
            time.sleep(1)
        """

        """ Draw the range of view of the robot

        for obstacle in self.obs_list:
            xx, yy = obstacle
            pygame.draw.rect(screen, (255, 255, 255), (xx * w, 1 + yy * h, w, h), 1)
            pygame.display.update()
            time.sleep(1)
        """
