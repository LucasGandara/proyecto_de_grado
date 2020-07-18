#!/usr/bin/env python   
# -*- coding: utf-8 -*-

import pygame
from pygame import QUIT
import numpy as np
import neat
import sys
import os
import time
from math import pow
import random
import visualize

WIDTH = 650
HEIGHT = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Training env")
clock = pygame.time.Clock()

# How many columns and rows?
cols = 32
rows = 32

#Width and height of each cell of grid
w = (WIDTH - 10) / cols
h = (HEIGHT - 10) / rows

class Spot(object):
    def __init__(self, x, y):
        self.f = 0
        self.g = 0
        self.h = 0
        self.x = x # posicion x del punto en el espacio
        self.y = y # Posicion y del punto en el espacio
        self.Neighbors = []
        self.Neighbors2 = []
        self.Neighbors3 = []
        self.previous = None
        self.obstacle = False
        self.color = pygame.Color(255, 255, 102)

    def addNeighbors(self, spots, n=1):
        if n == 1:
            if self.x >= 1:
                self.Neighbors.append(spots[self.x - 1][self.y])
            if self.x < (cols - 1):
                self.Neighbors.append(spots[self.x + 1][self.y])
            if self.y >= 1:
                self.Neighbors.append(spots[self.x][self.y - 1])
            if self.y < (rows - 1):
                self.Neighbors.append(spots[self.x][self.y + 1])
            if self.x > 0 and self.y > 0:
                self.Neighbors.append(spots[self.x - 1][self.y - 1])
            if self.x < cols - 1 and self.y > 0:
                self.Neighbors.append(spots[self.x + 1][self.y - 1])
            if self.x > 0 and self.y < rows - 1:
                self.Neighbors.append(spots[self.x - 1][self.y + 1])
            if self.x < cols - 1 and self.y < rows - 1:
                self.Neighbors.append(spots[self.x + 1][self.y + 1])

        if n == 2:
            if self.x >= 2:
                self.Neighbors2.append(spots[self.x - 2][self.y])
            if self.x < (cols - 2):
                self.Neighbors2.append(spots[self.x + 2][self.y])
            if self.y >= 2:
                self.Neighbors2.append(spots[self.x][self.y - 2])
            if self.y < (rows - 2):
                self.Neighbors2.append(spots[self.x][self.y + 2])
            if self.x > 2 and self.y > 2:
                self.Neighbors2.append(spots[self.x - 1][self.y - 2])
                self.Neighbors2.append(spots[self.x - 2][self.y - 1])
                self.Neighbors2.append(spots[self.x - 2][self.y - 2])
            if self.x < cols - 2 and self.y > 1:
                self.Neighbors2.append(spots[self.x + 2][self.y - 1])
                self.Neighbors2.append(spots[self.x + 2][self.y - 2])
                self.Neighbors2.append(spots[self.x + 1][self.y - 2])
            if self.x > 1 and self.y < rows - 2:
                self.Neighbors2.append(spots[self.x - 1][self.y + 2])
                self.Neighbors2.append(spots[self.x - 2][self.y + 1])
                self.Neighbors2.append(spots[self.x - 2][self.y + 2])
            if self.x < cols - 2 and self.y < rows - 2:
                self.Neighbors2.append(spots[self.x + 1][self.y + 2])
                self.Neighbors2.append(spots[self.x + 2][self.y + 2])
                self.Neighbors2.append(spots[self.x + 2][self.y + 1])

        if n == 3:
            if self.x >= 2:
                self.Neighbors3.append(spots[self.x - 3][self.y])
            if self.x < (cols - 3):
                self.Neighbors3.append(spots[self.x + 3][self.y])
            if self.y >= 2:
                self.Neighbors3.append(spots[self.x][self.y - 3])
            if self.y < (rows - 2):
                self.Neighbors3.append(spots[self.x][self.y + 3])
            if self.x > 3 and self.y > 3:
                self.Neighbors3.append(spots[self.x - 1][self.y - 3])
                self.Neighbors3.append(spots[self.x - 3][self.y - 1])
                self.Neighbors3.append(spots[self.x - 3][self.y - 2])
                self.Neighbors3.append(spots[self.x - 2][self.y - 3])
                self.Neighbors3.append(spots[self.x - 3][self.y - 3])
            if self.x < cols - 3 and self.y > 2:
                self.Neighbors3.append(spots[self.x + 3][self.y - 1])
                self.Neighbors3.append(spots[self.x + 3][self.y - 2])
                self.Neighbors3.append(spots[self.x + 1][self.y - 3])
                self.Neighbors3.append(spots[self.x + 2][self.y - 3])
                self.Neighbors3.append(spots[self.x + 3][self.y - 3])
            if self.x > 3 and self.y < rows - 3:
                self.Neighbors3.append(spots[self.x - 1][self.y + 3])
                self.Neighbors3.append(spots[self.x - 3][self.y + 1])
                self.Neighbors3.append(spots[self.x - 3][self.y + 2])
                self.Neighbors3.append(spots[self.x - 2][self.y + 3])
                self.Neighbors3.append(spots[self.x - 3][self.y + 3])
            if self.x < cols - 3 and self.y < rows - 3:
                self.Neighbors3.append(spots[self.x + 1][self.y + 3])
                self.Neighbors3.append(spots[self.x + 2][self.y + 3])
                self.Neighbors3.append(spots[self.x + 3][self.y + 1])
                self.Neighbors3.append(spots[self.x + 3][self.y + 2])
                self.Neighbors3.append(spots[self.x + 3][self.y + 3])

    def get_list_of_neigbors(self, n=1):
        """ Return the list of the neighbors on the given 
            index """
        
        Neighbors = []
        if n == 1:
            for neighbor in self.Neighbors:
                Neighbors.append([neighbor.x, neighbor.y])
        elif n == 2:
            for neighbor in self.Neighbors2:
                Neighbors.append([neighbor.x, neighbor.y])
        elif n == 3:
            for neighbor in self.Neighbors3:
                Neighbors.append([neighbor.x, neighbor.y])

        return Neighbors

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
            
    def view(self, obss_list):
        """ This function allows the virtual agent to watch it's surroundings"""
        self.obs         = []
        self.obs_list    = []
        self.obs_to_draw = []
        self.discrete_view = [-1 for i in range(363)]
        self.directional_view = [0, 0, 0, 0, 0, 0, 0, 0] 

        for i in range(0, self.range_of_view):
            for j in range(-i, i + 1):
                for k in range(-i, i + 1):
                    try:
                        if [self.x + j, self.y + k] not in self.obs_list: # Append only once
                            self.obs.append(spots[self.x + j][self.y + k])
                            self.obs_list.append([self.x + j, self.y + k])
                    except IndexError:
                        pass

        for x, obstacle in enumerate(self.obs_list):
            if obstacle in walls_list or obstacle in obss_list:
                self.obs_to_draw.append(obstacle)
                self.discrete_view[x] = 1

        self.discrete_view[-1] = start.y
        self.discrete_view[-2] = start.x

        directions = [ [self.x, self.y-1]  , [self.x+1, self.y-1], [self.x+1, self.y],
                       [self.x+1, self.y+1], [self.x, self.y+1]  , [self.x-1, self.y+1],
                       [self.x-1, self.y]  , [self.x-1, self.y-1] ]
        distances = [] 
        for direction in directions:
            distances.append(np.linalg.norm(np.array(direction) - np.array([end.x, end.y])))                        
        # Set the minimun to 1
        self.directional_view[distances.index(min(distances))] = 1

        for direction in self.directional_view:
            self.discrete_view.append(direction)

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
        
def redrawGameWindow(win, robots, endp, startp):
    pygame.draw.rect(win, (148, 148, 148), (0, 0, WIDTH, HEIGHT))

    #Dibujar la grilla
    for i in range(cols):
        for j in range(rows):
            pygame.draw.rect(win, (0, 0, 0), (spots[i][j].x * w, 1 + spots[i][j].y * h, w, h), 1)

    # Draw the medium risk Neighbors of the path
    #   for spot in path:
    #        for neighbour in spot.Neighbors2:
    #            pygame.draw.rect(win, (204, 230, 255), (neighbour.x * w + 2, 1 + neighbour.y * h + 2, w - 4, h - 4))


    # Draw the high risk Neighbors of the path
    #for spot in path:
    #    for neighbour in spot.Neighbors:
    #        pygame.draw.rect(win, (128, 191, 255), (neighbour.x * w + 2, 1 + neighbour.y * h + 2, w - 4, h- 4)) 
        
    # Draw the path to follow
    #for spot in path:
    #    pygame.draw.rect(win, spot.color, (spot.x * w, 1 + spot.y * h, w, h)) 
 
    # Draw the agent
    for agent in robots:
        pygame.draw.rect(win, agent.color, (agent.x * w, 1 + agent.y * h, w, h))
 
    # Draw the end spot!
    pygame.draw.rect(win, (255, 234, 0), (endp.x * w, 1 + endp.y * h, w, h))

    # Draw the start spot
    pygame.draw.rect(win, (0, 0, 0), (startp.x * w, 1 + startp.y * h, w, h))

    # Draw what the robot see
    if len(robots) > 1:
        for obstacle in robots[0].obs_to_draw:
            pygame.draw.rect(win, (255, 255, 255), (obstacle[0] * w, 1 + obstacle[1] * h, w, h))

    # Draw the wallls
    #for wall in walls:
    #    pygame.draw.rect(win, wall.color, (wall.x * w, 1 + wall.y * h, w - 1, h - 1))

    # Draw the Obstacles
    #for obstacle in obstacles:
    #    pygame.draw.rect(win, obstacle.color, (obstacle.x * w, 1 + obstacle.y * h, w - 1, h - 1))


    pygame.display.update()

# Create the 2D array
spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
for i in range(len(spots)):
    for j in range(len(spots[i])):
        spots[i][j].addNeighbors(spots, 1)

# Load the path to follow 
path_file = open('path2.txt', 'r')
path = []
tmp = path_file.read().split('\n')
path_file.close()
for pair in tmp:
    aux = pair.split()
    path.append(Spot(int(aux[0]), int(aux[1])))
    path[-1].addNeighbors(spots, n=1)
    path[-1].addNeighbors(spots, n=2)
    path[-1].color = pygame.Color(0, 0, 255)

# read the list of the wall and obstacles
wall_file = open('Wall.txt', 'r')
walls = []
walls_list = []
tmp = wall_file.read().split('\n')
wall_file.close()
for pair in tmp:
    aux = pair.split(',')
    walls.append(Spot(int(aux[0]), int(aux[1])))
    walls_list.append([int(aux[0]), int(aux[1])])
    walls[-1].color = pygame.Color(173, 0, 75)

# Add Start adn End point
randnum = np.random.randint(-9, 9)
randnum2 = np.random.randint(-6, 6)
start = spots[path[0].x + randnum2][path[0].y]
end   = spots[15 + randnum][28]

def main(genomes, config):
    nets = []
    ge = []
    robots = []

    obstacle_file = open('Obstacles.txt', 'r')
    obstacles = []
    obstacle_list = []
    tmp = obstacle_file.read().split('\n')
    obstacle_file.close()
    rand1 = random.randint(-4, 3)
    rand2 = random.randint(-4, 3)

    randnum3 = np.random.randint(-9, 9)
    randend   = spots[15 + randnum3][28]   
    randnum4 = np.random.randint(-6, 6)
    randstart = spots[path[0].x + randnum4][path[0].y]

    for pair in tmp:
        aux = pair.split(',')
        obstacles.append(Spot(int(aux[0]) - 3 + rand1, int(aux[1]) + rand1))
        obstacle_list.append([int(aux[0]) - 3 + rand1, int(aux[1]) + rand1])
        obstacles.append(Spot(int(aux[0]) + 8 + rand2, int(aux[1]) + 2 + rand2))
        obstacle_list.append([int(aux[0]) + 8 + rand2, int(aux[1]) + 2 + rand2])
    obstacle_file.close()

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        robots.append(Agent(randstart.x, path[0].y))
        g.fitness = 0
        ge.append(g)

    Gaming = True
    while Gaming:
        clock.tick(1)
        for events in pygame.event.get():
            pos = pygame.mouse.get_pos()
            if events.type == QUIT:
                sys.exit(0)

        if len(robots) <= 0:
            Gaming = False
            break

        for x, agent in enumerate(robots):
            agent.view(obstacle_list)
            output = nets[x].activate(agent.discrete_view)
            agent.go_to(output.index(max(output)))
            agent.lifepoints = agent.lifepoints - 1

            # Calculate the distance to final point
            agent.distToFinalPoint = np.linalg.norm(np.array([agent.x, agent.y]) - np.array([randend.x, randend.y]))

            # Calculate the distance to the closest point of the trajectory
            agent.closestPath = 100
            for spot in path:
                dist = np.linalg.norm(np.array([agent.x, agent.y]) - np.array([spot.x, spot.y]))
                if dist < agent.closestPath:
                    agent.closestPath = dist

            # If the robot crashes it dies and get a reward of 0
            for obstacle in obstacle_list:
                if obstacle == [agent.x, agent.y]:
                    ge[x].fitness = (100 / pow(agent.distToFinalPoint, 1.0/3)) - 15
                    robots.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue

            # If the robot crashes to a wall it dies and get a reward of 0
            for wall in walls:
                if [wall.x, wall.y] == [agent.x, agent.y]:
                    ge[x].fitness = (150 / pow(agent.distToFinalPoint, 1.0/3)) - 15
                    robots.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue

            # If the agent gets to the final point gets instant reward of 200
            if [agent.x, agent.y] == [randend.x, randend.y]:
                ge[x].fitness = 200
                print('a robot did it')
                #print(ge[x].fitness)
                robots.pop(x)
                nets.pop(x)
                ge.pop(x)
                continue

            # If the agent have 0 life points calculate the current fitness and multiply it by 0.7
            if agent.lifepoints <= 0:
                ge[x].fitness = 150 / pow(agent.distToFinalPoint, 1.0/3)
                #print(ge[x].fitness)
                robots.pop(x)
                nets.pop(x)
                ge.pop(x)
                continue
            
        if len(robots) <= 3:
            redrawGameWindow(screen, robots, randend, randstart)
        else:
            redrawGameWindow(screen, robots, randend, randstart)

def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-628')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.checkpoint.Checkpointer(generation_interval=20,
                                                time_interval_seconds=1000,
                                                filename_prefix='neat-checkpoint-'))
    winner = p.run(main, 100)

    visualize.draw_net(config, winner, True, node_names=None)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    run(config_path)
