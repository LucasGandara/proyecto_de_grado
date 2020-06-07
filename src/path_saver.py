#!/usr/bin/env python   
# -*- coding: utf-8 -*-
"""
States:
    Open set: nodes that still needs to be evaluated
    closed set: all the nodes that have finished been evaluated
"""
import pygame
from pygame import QUIT
import sys
from math import  hypot
from os import system
from datetime import datetime
import random
from random import randint
pygame.init()
#first seed = 3
#random.seed(3)
#Second seed = 35
#random.seed(35)
#Third seed
#random.seed(10)
# Fourth seed
#random.seed(15)
# Fifth seed
random.seed(9)

WIDTH = 650
HEIGHT = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Algorithm")
clock = pygame.time.Clock()
obstacles = []

# How many columns and rows?
cols = 32
rows = 32

#Width and height of each cell of grid
w = (WIDTH - 10) / cols
h = (HEIGHT - 10) / rows
path = []
current = []
class Spot(object):
    def __init__(self, x, y):
        self.f = 0
        self.g = 0
        self.h = 0
        self.x = x # posicion x del punto en el espacio
        self.y = y # Posicion y del punto en el espacio
        self.Neighbors = []
        self.previous = None
        self.obstacle = False

    def addNeighbors(self, spots):
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

    def Isover(self):
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
        return False

def heuristic(a, b):
    return hypot(a.x - b.x, a.y - b.y)

def redrawGameWindow(win):
    #pygame.draw.rect(win, (255, 255, 255), (0, 0, WIDTH, HEIGHT))

    #Dibujar la grilla
    for i in range(cols):
        for j in range(rows):
            pygame.draw.rect(win, (0, 0, 255), (spots[i][j].x * w, 1 + spots[i][j].y * h, w, h), 1)
    
    # Dibujamos los openSet con verde
    for i,spot in enumerate(OpenSet):
        pygame.draw.rect(win, (0, 255, 0), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    #We draw ClosedSet with red
    for i,spot in enumerate(closedSet):
        pygame.draw.rect(win, (255, 0, 0), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))
   
    # Draw current
    pygame.draw.rect(win, (255, 0, 255), (2 + current.x * w, 3 + current.y * h, w - 4, h - 4))

   # Draw the path in blue
    for spot in path:
        pygame.draw.rect(win, (0, 0, 255), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    # Draw Obstacles
    for spot in obstacles:
        pygame.draw.rect(win, (255, 255, 102), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    # Draw the walls
    for wall in walls:
        pygame.draw.rect(win, (255, 0, 0), (2 + wall.x * w, 3 + wall.y * h, w - 4, h - 4))

    # Draw start and end spot!
    pygame.draw.rect(win, (255, 255, 255), (start.x * w, 1 + start.y * h, w, h))
    pygame.draw.rect(win, (198, 252, 3), (end.x * w, 1 + end.y * h, w, h))

    pygame.display.update()

# Create the 2D array
spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
for i in range(len(spots)):
    for j in range(len(spots[i])):
        spots[i][j].addNeighbors(spots)

# Definir Obstáculos
obstalce_x = []
obstacle_y = []
walls = []
obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Wall.txt')
pair_coodinates = obstacle_file.read()
obstacle_list = pair_coodinates.split('\n')
for pair in obstacle_list:
    pair2 = pair.split(',')
    obstalce_x.append(int(pair2[0]))
    obstacle_y.append(int(pair2[1]))

for x, y in zip(obstalce_x, obstacle_y):
    spots[int(x)][int(y)].obstacle = True
    walls.append(spots[int(x)][int(y)])

obstalce_x = []
obstacle_y = []
obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt')
pair_coodinates = obstacle_file.read()
obstacle_list = pair_coodinates.split('\n')
for pair in obstacle_list:
    pair2 = pair.split(',')
    obstalce_x.append(int(pair2[0]))
    obstacle_y.append(int(pair2[1]))

rand1 = randint(-4, 4)
rand2 = randint(-4, 3)

for x, y in zip(obstalce_x, obstacle_y):
    spots[int(x - 3) + rand1][int(y) + rand1].obstacle = True
    obstacles.append(spots[int(x - 3) + rand1][int(y) + rand1])
    spots[int(x + 8) + rand2][int(y + 2) + rand2].obstacle = True
    obstacles.append(spots[int(x + 8) + rand2][int(y + 2) + rand2])

OpenSet = []
closedSet = []

start = spots[25 + randint(-3, 3)][5]
end = spots[4 + randint(-3, 3)][28]
path = []
OpenSet.append(start)
current = start

gaming = True
while gaming:
    clock.tick(12)
    #Find thepath
    temp = current
    path = []
    path.append(temp)
    #As long as the temp has a previous
    while temp.previous:
        current = temp
        path.append(temp.previous)
        temp = temp.previous
    for eventos in pygame.event.get():
        pos = pygame.mouse.get_pos()
        if eventos.type == QUIT:
            sys.exit(0)
    
    # Find the one to evaluate next
    if len(OpenSet) > 0:
        winner = 0

        for i in range(len(OpenSet)):
            if OpenSet[i].f < OpenSet[winner].f:
                winner = i

            if OpenSet[i].f == OpenSet[winner].f:
                if OpenSet[i].g > OpenSet[winner].g:
                    winner = i

        current = OpenSet[winner]
        lastCheckedNode = current

        if current == end:
            #Find the path
            path = []
            temp = current
            path.append(temp)
            #As long as the temp has a previous
            pygame.image.save(screen, '../training_envs/env_5.png')
            while temp.previous:
                current = temp
                path.append(temp.previous)
                temp = temp.previous
            del(path[-1])
            del(path[-1])
            system('cls')
            print('Finish!')
            gaming = False 
        try:
            OpenSet.remove(current)
        except ValueError as e:
            pass
        
        closedSet.append(current)
    
        # Verify Neighbors of the current cell
        neighbors = current.Neighbors
        for neighbor in neighbors:
            if not(neighbor in closedSet)  and not(neighbor.obstacle): # ceck if neighbor is available to visit
                temp = current.g + heuristic(neighbor, current)

                newpath = False

                if not(neighbor in OpenSet):
                    OpenSet.append(neighbor)
                elif temp >= neighbor.g:
                        continue

                neighbor.g = temp
                neighbor.h = heuristic(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h

                neighbor.previous = current                
            
    else:
        # No solution
        print('No Solution')
        gaming = False
        pass

    redrawGameWindow(screen)

pygame.image.save(screen, '/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba' + datetime.now().strftime('%H_%M_%S') + '.png')

what_to_save = 'cells'

if what_to_save == 'cells':
    """ Once the Path finder algoritm ends, export the path for the turtlebot to follow """
    X_references = []
    Y_references = []
    rpath = list(reversed(path))
    path_file = open('path5.txt', 'w')
    for spot in rpath:
        #X_references.append((spot.y - 3) * 0.178)
        #Y_references.append((spot.x - 28) * 0.178)
        if spot != rpath[-1]:
            path_file.write('%s %s\n' % (spot.x, spot.y))
        else:
            path_file.write('%s %s' % (spot.x, spot.y))
elif what_to_save == 'real':
    # Save the real positions
    path_file = open('path.txt', 'w')
    for x, y in zip(X_references, Y_references):
        if x != X_references[-1]:
            path_file.write('%s %s\n'% (x, y))
        else:
            path_file.write('%s %s'% (x, y))
    path_file.close()
