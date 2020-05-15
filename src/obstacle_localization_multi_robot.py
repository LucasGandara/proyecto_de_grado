#!/usr/bin/env python   
# -*- coding: utf-8 -*-

import pygame
from pygame.locals import QUIT
import sys
from datetime import datetime
from math import  hypot
from os import system
import subprocess
pygame.init()

WIDTH = 650
HEIGHT = 500
path_screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Algorithm")
clock = pygame.time.Clock()
walls = []
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

class A_star():

    def __init__(self):
        rospy.init_node('path_and_localization', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('/main_tb3/cmd_vel', Twist, queue_size=5)
        position = Point()
        move_cmd = Twist()
        r = rospy.Rate(10)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = '/main_tb3/odom'
        localization_screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Localization Algorithm")
        self.temp_detected_obstacles = []
        self.detected_obstacles = [Spot(0, 0)]
        self.detected_obstacles_list = []
        self.new_obstacles = []
        self.burger_orientation = [0, 0, 0, 0]
        self.TETA = 0
        self.laser = [0 for i in range(360)]
        self.robot_track = [] # Positions where the robot have been
        self.robot_track_spots = [] # Same positions but in Spot form
        rospy.Subscriber('/main_tb3/scan', LaserScan, self.get_laser, queue_size=1)
        rospy.Subscriber("/main_tb3/odom", Odometry , self.get_theta, queue_size=1)        

        try:
            self.tf_listener.waitForTransform(self.odom_frame, '/main_tb3/base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = '/main_tb3/base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, '/main_tb3/base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = '/main_tb3/base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        
        for i in range(len(X_references)):

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

    def redrawGameWindow(self, win):
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
            pygame.draw.rect(win, (222, 80, 80), (obstacle.x * w + 2, 3 + obstacle.y * h, (w - 3), (h - 3)), 0)
        """
        for obstacle in self.detected_obstacles_list:
            pygame.draw.rect(win, (255, 255, 102), (obstacle[0] * w + 2, 3 + obstacle[1] * h, (w - 3), (h - 3)), 0)            

        for obstacle in self.new_obstacles:
            pygame.draw.rect(win, (222, 80, 80), (obstacle[0] * w + 2, 3 + obstacle[1] * h, (w - 3), (h - 3)), 0)
        """ 
        pygame.display.update()

A_star()