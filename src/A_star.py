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
pygame.init()

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

    # Draw Obstacles
    for spot in obstacles:
        pygame.draw.rect(win, (255, 255, 102), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    # Draw start and end spot!
    pygame.draw.rect(win, (255, 255, 255), (start.x * w, 1 + start.y * h, w, h))
    pygame.draw.rect(win, (0, 204, 0), (end.x * w, 1 + end.y * h, w, h))

    pygame.display.update()

# Create the 2D array
spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
for i in range(len(spots)):
    for j in range(len(spots[i])):
        spots[i][j].addNeighbors(spots)

# Definir Obstáculos
obstalce_x = []
obstacle_y = []
obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt')
pair_coodinates = obstacle_file.read()
obstacle_list = pair_coodinates.split('\n')
for pair in obstacle_list:
    pair2 = pair.split(',')
    obstalce_x.append(int(pair2[0]))
    obstacle_y.append(int(pair2[1]))
print(obstalce_x, obstacle_y)

for x, y in zip(obstalce_x, obstacle_y):
    spots[int(x)][int(y)].obstacle = True
    obstacles.append(spots[int(x)][int(y)])

OpenSet = []
closedSet = []

start = spots[15][12]
end = spots[4][28]
OpenSet.append(start)
current = start

gaming = True
while gaming:
    clock.tick(12)
    #Find the path
    path = []
    temp = current
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
                
        current = OpenSet[winner] 
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
            system('cls')
            print('Finish!')
            gaming = False
            pygame.image.save(screen, '/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/screenshot.png')
        try:
            OpenSet.remove(current)
        except ValueError as e:
            pass
        
        closedSet.append(current)
    
        # Verify Neighbors of the current cell
        neighbors = current.Neighbors
        for neighbor in neighbors:
            if not(neighbor in closedSet)  and not(neighbor.obstacle): # ceck if neighbor is available to visit
                temp = current.g + 1

                if neighbor in OpenSet:
                    if temp < neighbor.g:
                        neighbor.g = temp
                else:
                    newpath = True
                    neighbor.g = temp
                    OpenSet.append(neighbor)
        
                neighbor.previous = current
            # We aply Heuristics
            if newpath:
                neighbor.h = heuristic(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h
            
    else:
        # No solution
        print('No Solution')
        gaming = False
        pass

    redrawGameWindow(screen)


""" Once the Path finder algoritm ends, export the path for the turtlebot to follow """
X_references = []
Y_references = []
for spot in path:
    X_references.append((spot.y - start.y) * 0.178)
    Y_references.append((spot.x - start.x) * 0.178)

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import numpy as np

class GotoPoint():
    def __init__(self):
        rospy.init_node('A_Star_Path_Pinder', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        position = Point()
        move_cmd = Twist()
        r = rospy.Rate(10)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

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
        
        for i in range(len(X_references)):
            (position, rotation) = self.get_odom()
            last_rotation = 0
            linear_speed = 1
            angular_speed = 1
            goal_x = X_references[i]
            goal_y = Y_references[i]
            goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
            distance = goal_distance

            while distance > 0.05:
                (position, rotation) = self.get_odom()
                x_start = position.x
                y_start = position.y
                path_angle = atan2(goal_y - y_start, goal_x- x_start)

                if path_angle < -pi/4 or path_angle > pi/4:
                    if goal_y < 0 and y_start < goal_y:
                        path_angle = -2*pi + path_angle
                    elif goal_y >= 0 and y_start > goal_y:
                        path_angle = 2*pi + path_angle
                if last_rotation > pi-0.1 and rotation <= 0:
                    rotation = 2*pi + rotation
                elif last_rotation < -pi+0.1 and rotation > 0:
                    rotation = -2*pi + rotation
                move_cmd.angular.z = angular_speed * path_angle-rotation

                distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
                move_cmd.linear.x = min(linear_speed * distance, 0.1)

                if move_cmd.angular.z > 0:
                    move_cmd.angular.z = min(move_cmd.angular.z, 1.5)
                else:
                    move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

                last_rotation = rotation
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            (position, rotation) = self.get_odom()

        self.cmd_vel.publish(Twist())

    def getkey(self):
        x, y, z = raw_input("| x | y | z |\n").split()
        if x == 's':
            self.shutdown()
        x, y, z = [float(x), float(y), float(z)]
        return x, y, z

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])

    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

GotoPoint()