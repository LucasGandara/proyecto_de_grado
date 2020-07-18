#!/usr/bin/env python   
# -*- coding: utf-8 -*-

import pygame
from pygame.locals import QUIT
import sys
from datetime import datetime
from math import  hypot
import numpy as np
import os
import random
import subprocess
import neat
pygame.init()

# How many columns and rows?
cols = 32
rows = 32
""" Load NEAT """
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
        if n == 0: # Stay still pass
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

# Create the 2D array
spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
for i in range(len(spots)):
    for j in range(len(spots[i])):
        spots[i][j].addNeighbors(spots, 1)

# Load the path to follow 
path_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/path2.txt', 'r')
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
wall_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Wall.txt', 'r')
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
    done_flag = False

    obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt', 'r')
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
                if not done_flag:
                    print('a robot did it')
                    done_flag = True
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

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

print 'Reading neat checkpoint'
p = neat.Checkpointer.restore_checkpoint('/home/lucas/catkin_ws/src/proyecto_de_grado/src/neat-checkpoint-628')
print 'neat Checkpoint read'

print 'Looking for the best genome'
winner = p.run(main, 1)
print 'Best genome finded'

"""Finish load NEAT """

WIDTH = 650
HEIGHT = 500
path_screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Algorithm")
clock = pygame.time.Clock()
walls = []
obstacles = []

#Width and height of each cell of grid
w = (WIDTH - 10) / cols
h = (HEIGHT - 10) / rows
path = []
current = []

def heuristic(a, b):
    return hypot(a.x - b.x, a.y - b.y)

def redrawGameWindow(win):
    #pygame.draw.rect(win, (255, 255, 255), (0, 0, WIDTH, HEIGHT))

    #Dibujar la grilla
    for i in range(cols):
        for j in range(rows):
            pygame.draw.rect(win, (0, 0, 255), (spots[i][j].x * w, 1 + spots[i][j].y * h, w, h), 1)
    
    # Dibujams los openSet con verde
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

    # Draw Wall
    for spot in walls:
        pygame.draw.rect(win, (255, 255, 102), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    # Draw obstacles
    for spot in obstacles:
        pygame.draw.rect(win, (255, 255, 40), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    # Draw start and end spot!
    pygame.draw.rect(win, (255, 255, 255), (start.x * w, 1 + start.y * h, w, h))
    pygame.draw.rect(win, (198, 252, 3), (end.x * w, 1 + end.y * h, w, h))

    pygame.display.update()

# Create the 2D array
spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
for i in range(len(spots)):
    for j in range(len(spots[i])):
        spots[i][j].addNeighbors(spots)

# Definir donde están las paredes
obstalce_x = []
obstacle_y = []
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

# Definir donde están las paredes
obstalce_x = []
obstacle_y = []
obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt')
pair_coodinates = obstacle_file.read()
obstacle_list = pair_coodinates.split('\n')
for pair in obstacle_list:
    pair2 = pair.split(',')
    #obstalce_x.append(int(pair2[0]) - 6)
    #obstacle_y.append(int(pair2[1]))
    obstalce_x.append(int(pair2[0]))
    obstacle_y.append(int(pair2[1]))

obstacle_list = []
for x, y in zip(obstalce_x, obstacle_y):
    spots[int(x)][int(y)].obstacle = True
    obstacles.append(spots[int(x)][int(y)])
    obstacle_list.append([x, y])

    """States:
        Open set: nodes that still needs to be evaluated
        closed set: all the nodes that have finished been evaluated"""

OpenSet = []
closedSet = []

start = spots[29][3]
end = spots[3][28]
path = []
OpenSet.append(start)
current = start

gaming = True
while gaming:
    clock.tick(12)
    #Find the path
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
            while temp.previous:
                current = temp
                path.append(temp.previous)
                temp = temp.previous
            del(path[-1])
            os.system('cls')
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

    redrawGameWindow(path_screen)

pygame.image.save(path_screen, '/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba2.png')

# Show saved image
subprocess.call(['xdg-open', '/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba2.png'])

""" Once the Path finder algoritm ends, export the path for the turtlebot to follow """
X_references = []
Y_references = []
paht_list = []
neighbors_list_high_risk = []
neighbors_list_medium_risk = []
neighbors_list_low_risk = []

for spot in reversed(path):
    X_references.append((spot.y - start.y) * 0.178)
    Y_references.append((spot.x - start.x) * 0.178)
    paht_list.append([spot.x, spot.y])

    spot.addNeighbors(spots)
    spot.addNeighbors(spots, n=2)
    spot.addNeighbors(spots, n=3)

    aux = spot.get_list_of_neigbors(n=1)
    for neigh in aux:
        neighbors_list_high_risk.append(neigh)

    aux = spot.get_list_of_neigbors(n=2)
    for neigh in aux:
        neighbors_list_medium_risk.append(neigh)

    aux = spot.get_list_of_neigbors(n=3)
    for neigh in aux:
        neighbors_list_low_risk.append(neigh)

x_followed = []
y_followed = []

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf
from math import radians, copysign, sqrt, pow, pi, atan2, sin, cos
from tf.transformations import euler_from_quaternion
import numpy as np
import matplotlib.pyplot as plt
import neat

class A_star():

    def __init__(self):
        rospy.init_node('path_and_localization', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        position = Point()
        move_cmd = Twist()
        r = rospy.Rate(10)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'
        localization_screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Localization Algorithm")
        self.temp_detected_obstacles = []
        self.detected_obstacles = [Spot(0, 0)]
        self.detected_obstacles_list = []
        self.new_obstacles = []
        self.burger_orientation = [0, 0, 0, 0]
        self.TETA = 0
        self.laser = [0 for i in range(360)]
        self.control_flag = 'A_star'
        self.robot_track = [] # Positions where the robot have been
        self.robot_track_spots = [] # Same positions but in Spot form
        rospy.Subscriber('/scan', LaserScan, self.get_laser, queue_size=1)
        rospy.Subscriber("/odom", Odometry , self.get_theta, queue_size=1)        

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        
        rospy.loginfo('A* on control')

        for i in range(len(X_references)):
            if self.control_flag == 'A_star':
                pass
            elif self.control_flag == 'NEAT':
                break
            #print """----------------------------------
            #Going To point:
            #%s, %s
            #""" % (X_references[i], Y_references[i])

            (position, orientation) = self.get_odom()
        
            last_rotation = 0
            linear_speed = 1
            angular_speed = 1
            goal_x = X_references[i]
            goal_y = Y_references[i]
            goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
            distance = goal_distance

            while distance > 0.05:

                x_followed.append(position.y)
                y_followed.append(-1 * position.x)

                (position, orientation) = self.get_odom()
                x_start = position.x
                y_start = position.y
                errror_teta = atan2(goal_y - y_start, goal_x- x_start)

                move_cmd.angular.z = angular_speed * errror_teta-orientation

                distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
                move_cmd.linear.x = min(linear_speed * distance, 0.1)

                if move_cmd.angular.z > 0:
                    move_cmd.angular.z = min(move_cmd.angular.z, 1.5)
                else:
                    move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

                last_rotation = orientation
                self.cmd_vel.publish(move_cmd)
                r.sleep()

                self.redrawGameWindow(localization_screen)

            (position, orientation) = self.get_odom()

        if self.control_flag == 'NEAT':
            rospy.loginfo('NEAT Taking control now')
            self.cmd_vel.publish(Twist())

            # Create the agent
            xrobot = start.x + (position.y // 0.178)
            yrobot = start.y + (position.x // 0.178)
            agent = Agent(int(xrobot), int(yrobot))
            
            # Create the brain of the agent
            best_genome = p.best_genome
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         '/home/lucas/catkin_ws/src/proyecto_de_grado/src/config.txt')

            agentBrain = neat.nn.FeedForwardNetwork.create(best_genome, config)

            for i in range(15):
                agent.view(self.temp_detected_obstacles)

                output = agentBrain.activate(agent.discrete_view)
                agent.go_to(output.index(max(output)))


                goal_x = (agent.y - start.y) * 0.178
                goal_y = (agent.x - start.x) * 0.178

                (position, orientation) = self.get_odom()
        
                last_rotation = 0
                linear_speed = 1
                angular_speed = 1
                goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
                distance = goal_distance

                while distance > 0.05:

                    x_followed.append(position.y)
                    y_followed.append(-1 * position.x)

                    (position, orientation) = self.get_odom()
                    x_start = position.x
                    y_start = position.y
                    errror_teta = atan2(goal_y - y_start, goal_x- x_start)

                    move_cmd.angular.z = angular_speed * errror_teta-orientation

                    distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
                    move_cmd.linear.x = min(linear_speed * distance, 0.1)

                    if move_cmd.angular.z > 0:
                        move_cmd.angular.z = min(move_cmd.angular.z, 1.5)
                    else:
                        move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

                    last_rotation = orientation
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()

                    self.redrawGameWindow(localization_screen)

            (position, orientation) = self.get_odom()
        
            last_rotation = 0
            linear_speed = 1
            angular_speed = 1
            goal_x = X_references[-1]
            goal_y = Y_references[-1]
            goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
            distance = goal_distance

            while distance > 0.05:

                x_followed.append(position.y)
                y_followed.append(-1 * position.x)

                (position, orientation) = self.get_odom()
                x_start = position.x
                y_start = position.y
                errror_teta = atan2(goal_y - y_start, goal_x- x_start)

                move_cmd.angular.z = angular_speed * errror_teta-orientation

                distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
                move_cmd.linear.x = min(linear_speed * distance, 0.1)

                if move_cmd.angular.z > 0:
                    move_cmd.angular.z = min(move_cmd.angular.z, 1.5)
                else:
                    move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

                last_rotation = orientation
                self.cmd_vel.publish(move_cmd)
                r.sleep()

                self.redrawGameWindow(localization_screen)


            self.redrawGameWindow(localization_screen, agent)

        # Save the final robot track
        now = datetime.now().strftime("%H:%M:%S").replace(':', '_')
        pygame.image.save(path_screen, '/home/lucas/catkin_ws/src/proyecto_de_grado/detection_trials/full_robot_track_at_' + now + '.png')

        self.cmd_vel.publish(Twist())
        fig = plt.figure()
        plt.plot(x_followed, y_followed)
        plt.title('Prueba 1')
        fig.suptitle('desplazamiento del robot durante la ejecucion', fontsize=20)
        plt.xlabel('X - Coordinates', fontsize=18)
        plt.ylabel('Y - Coordinates', fontsize=18)
        ax = plt.gca()
        ax.set_ylim([-4.984, 0.534])
        ax.set_xlim([-5.162, 0.356])
        plt.show()
        fig.savefig('/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba1.png')

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])

    def get_laser(self, msg):
        self.laser = msg.ranges

    def get_theta(self, msg):
        """ This function its only to calculate the orientation of 
            the robot in degrees"""
        self.burger_orientation[0] = msg.pose.pose.orientation.x
        self.burger_orientation[1] = msg.pose.pose.orientation.y
        self.burger_orientation[2] = msg.pose.pose.orientation.z
        self.burger_orientation[3] = msg.pose.pose.orientation.w

        euler = euler_from_quaternion(self.burger_orientation)
        self.TETA = euler[2]
        #rospy.loginfo('Orientacion actual: %s' % np.rad2deg(self.TETA))

    def shutdown(self):
        """ When the node closes, stop the robot"""
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def redrawGameWindow(self, win, agent=None):
        risk_flag = False
        pygame.draw.rect(win, (0, 0, 0), (0, 0, WIDTH, HEIGHT))

        (position, _) = self.get_odom()

        #Dibujar la grilla
        for i in range(cols):
            for j in range(rows):
                pygame.draw.rect(win, (0, 0, 255), (spots[i][j].x * w, 1 + spots[i][j].y * h, w, h), 1)

        # Draw the preview of the path with the risks zones
        """
        for spot in path:
            for neighbor in spot.Neighbors3:
                pygame.draw.rect(win, (255, 255, 255), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)

        for spot in path:
            for neighbor in spot.Neighbors2:
                pygame.draw.rect(win, (170, 255, 170), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)
        
        for spot in path:
            for neighbor in spot.Neighbors:
                pygame.draw.rect(win, (100, 255, 100), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)

        for spot in path:
            pygame.draw.rect(win, (0, 255, 0), (spot.x * w + 2, 3 + spot.y * h, (w - 3), (h - 3)), 0)
        """
        # Draw the actual position in grid of the robot
        # xr: actual position x of the robot in the grid; yr: actual position y of the robot in the grid
        xr = start.x + (position.y // 0.178)
        yr = start.y + (position.x // 0.178)
        point2 = [xr, yr]
        if not point2 in self.robot_track: # only append a point once
            self.robot_track.append(point2)
            self.robot_track_spots.append(Spot(int(point2[0]), int(point2[1])))
            self.robot_track_spots[-1].addNeighbors(spots)
            self.robot_track_spots[-1].addNeighbors(spots, n=2)
            self.robot_track_spots[-1].addNeighbors(spots, n=3)
    
       # Draws the risk zone of the positions of the robot
        """
        for spot in self.robot_track_spots:
            for neighbor in spot.Neighbors3:
                pygame.draw.rect(win, (255, 255, 255), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)

        for spot in self.robot_track_spots:
            for neighbor in spot.Neighbors2:
                pygame.draw.rect(win, (170, 255, 170), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)

        for spot in self.robot_track_spots:
            for neighbor in spot.Neighbors:
                pygame.draw.rect(win, (100, 255, 100), (neighbor.x * w + 2, 3 + neighbor.y * h, (w - 3), (h - 3)), 0)
        """ 
        self.temp_detected_obstacles =  []
        for point in self.robot_track:
            if point == self.robot_track[-1]:
                pygame.draw.rect(win, (255, 0, 255), (point[0] * w + 2, 3 + point[1] * h, (w - 3), (h - 3)), 0)
            else:
                pygame.draw.rect(win, (0, 255, 0), (point[0] * w + 2, 3 + point[1] * h, (w - 3), (h - 3)), 0)

        # Get the position of the obstacles
        # xo = position x of the obstacle, yo = position y of the obstacle
        for i, laser_distance in enumerate(self.laser):
            if laser_distance <= 2 and laser_distance >= 0.12:

                xo = position.x + laser_distance * cos(self.TETA + np.deg2rad(i))
                yo = position.y + laser_distance * sin(self.TETA + np.deg2rad(i))

                xo_in_grid = start.x + (yo // 0.178)
                yo_in_grid = start.y + (xo // 0.178)
                
                self.temp_detected_obstacles.append(Spot(xo_in_grid, yo_in_grid))
                if not [xo_in_grid, yo_in_grid] in self.detected_obstacles_list:
                    self.detected_obstacles_list.append([xo_in_grid, yo_in_grid])
                    self.detected_obstacles.append(Spot(xo_in_grid, yo_in_grid))

        for obstacle in self.detected_obstacles_list:
            if obstacle in obstacle_list: # Obstáculos predefinidos
                pass
            else:
                if not obstacle in self.new_obstacles:
                    self.new_obstacles.append(obstacle)
                    risk_flag = True

        # IF the obstacle is in the way, print an alert
        for obstacle in self.new_obstacles:
            if risk_flag:
                if obstacle in paht_list:
                    rospy.loginfo('Imminent Crash!!')
                    risk_flag = False
                    if self.control_flag == 'A_star':
                        self.control_flag = 'NEAT'

        for obstacle in self.new_obstacles:
            if risk_flag:
                if obstacle in neighbors_list_high_risk:
                        rospy.loginfo('High risk of Crash!!')
                        risk_flag = False

        for obstacle in self.new_obstacles:          
            if risk_flag:
                if obstacle in neighbors_list_medium_risk:
                    rospy.loginfo('Medium risk of Crash!!')
                    risk_flag = False
        
        for obstacle in self.new_obstacles:
            if risk_flag:
                if obstacle in neighbors_list_low_risk:
                    rospy.loginfo('low risk of Crash!!')
                    risk_flag = False

        # Draw the obstacles detected
        for obstacle in self.temp_detected_obstacles:
            pygame.draw.rect(win, (255, 255, 255), (obstacle.x * w + 2, 3 + obstacle.y * h, (w - 3), (h - 3)), 0)
        """
        for obstacle in self.detected_obstacles_list:
            pygame.draw.rect(win, (255, 255, 102), (obstacle[0] * w + 2, 3 + obstacle[1] * h, (w - 3), (h - 3)), 0)            

        for obstacle in self.new_obstacles:
            pygame.draw.rect(win, (222, 80, 80), (obstacle[0] * w + 2, 3 + obstacle[1] * h, (w - 3), (h - 3)), 0)
        """ 

        # When neat takes control draw the new agent
        if agent:
            pygame.draw.rect(win, agent.color, (agent.x * w + 2, 3 + agent.y * h, w - 3, h - 3), 0)

        pygame.display.update()

A_star()
