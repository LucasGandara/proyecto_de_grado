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

    # Dibujams los openSet con verde
    for i,spot in enumerate(OpenSet):
        pygame.draw.rect(win, (0, 255, 0), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

    #We draw ClosedSet with red
    for i,spot in enumerate(closedSet):
        pygame.draw.rect(win, (255, 0, 0), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))
   
    # Draw current
    pygame.draw.rect(win, (255, 0, 255), (2 + current.x * w, 3 + current.y * h, w - 4, h - 4))

    # Draw the path in blue
    for spot in total_path:
        pygame.draw.rect(win, (0, 0, 255), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))    

    # Draw Obstacles
    for spot in obstacles:
        pygame.draw.rect(win, (255, 255, 102), (2 + spot.x * w, 3 + spot.y * h, w - 4, h - 4))

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
obstacle_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/Obstacles.txt')
pair_coodinates = obstacle_file.read()
obstacle_list = pair_coodinates.split('\n')
for pair in obstacle_list:
    pair2 = pair.split(',')
    obstalce_x.append(int(pair2[0]))
    obstacle_y.append(int(pair2[1]))

for x, y in zip(obstalce_x, obstacle_y):
    spots[int(x)][int(y)].obstacle = True
    obstacles.append(spots[int(x)][int(y)])

X_references = []
Y_references = []
points_to_go = [(21, 20), (28, 28),(2, 28), (13, 6), (26,8)]
path = []
total_path = []
direction_changed = []
z = 0
z_changed = True

gaming = True
while gaming:
    if z_changed:
        OpenSet = []
        closedSet = []
        start = spots[points_to_go[z][0]][points_to_go[z][1]]
        end = spots[points_to_go[z + 1][0]][points_to_go[z + 1][1]]
        OpenSet.append(start)
        current = start
        z_changed = False
        spots = [[Spot(i, j) for j in range(rows)] for i in range(cols)]
        for i in range(len(spots)):
            for j in range(len(spots[i])):
                spots[i][j].addNeighbors(spots)
        for x, y in zip(obstalce_x, obstacle_y):
            spots[int(x)][int(y)].obstacle = True
            obstacles.append(spots[int(x)][int(y)])

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
                total_path.append(temp.previous)
                temp = temp.previous
            for spot in reversed(path):
                    X_references.append((spot.y - points_to_go[0][1]) * 0.178)
                    Y_references.append((spot.x - points_to_go[0][0]) * 0.178)
            direction_changed.append(len(X_references))
            print('Finish!')
            z += 1
            z_changed = True
            if z == len(points_to_go) - 1:
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

pygame.image.save(screen, '/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba2.png')

""" Once the Path finder algoritm ends, export the path for the turtlebot to follow """
x_followed = []
y_followed = []
goal_zs = []
from math import atan2
for i in range(len(X_references) - 1):
    goal_zs.append(atan2(Y_references[i + 1] - Y_references[i],X_references[i + 1] - X_references[i])) 
goal_zs.append(goal_zs[-1])

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi
from tf.transformations import euler_from_quaternion
import numpy as np
import matplotlib.pyplot as plt

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
            
            x_followed.append(position.y)
            y_followed.append(-1 * position.x)

            last_rotation = 0
            linear_speed = 1
            angular_speed = 1
            goal_x = X_references[i]
            goal_y = Y_references[i]
            goal_z = goal_zs[i]
            goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
            distance = goal_distance
            changepathflag = 0

            print """------------------------------
            siguiente punto:
            %s, %s
            """ % (goal_x, goal_y)

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

            while abs(rotation - goal_z) > 0.016:
                (position, rotation) = self.get_odom()
                if goal_z >= 0:
                    if rotation <= goal_z and rotation >= goal_z - pi:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                else:
                    if rotation <= goal_z + pi and rotation > goal_z:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                self.cmd_vel.publish(move_cmd)
                r.sleep()

        self.cmd_vel.publish(Twist())
        fig = plt.figure()
        plt.plot(x_followed, y_followed)
        plt.title('Prueba 1')
        fig.suptitle('desplazamiento del robot durante la ejecucion', fontsize=20)
        plt.xlabel('X - Coordinates', fontsize=18)
        plt.ylabel('Y - Coordinates', fontsize=18)
        ax = plt.gca()
        #ax.set_ylim([0, 5.7])
        #ax.set_xlim([0, 5.7])
        plt.show()
        fig.savefig('/home/lucas/catkin_ws/src/proyecto_de_grado/Imgs/Prueba1.png')

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