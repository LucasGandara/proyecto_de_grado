#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np 
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf
from tf.transformations import euler_from_quaternion
from math import radians, copysign, sqrt, atan2, sin, cos
from time import time
import neat

#start = [3][3]
#end   = [28][28]

# Load the path file 
path_file = open('/home/lucas/catkin_ws/src/proyecto_de_grado/src/path.txt')
pair_coordinates = path_file.read().split('\n')
path_x = []
path_y = []
for pair in pair_coordinates:
    aux = pair.split()
    path_x.append(float(aux[0]))
    path_y.append(float(aux[1]))

original_path_x = np.array(path_x)
original_path_y = np.array(path_y)

first_tb3_path_x = original_path_x + 9.5
first_tb3_path_y = original_path_y - 9.5

# Initialize ROS
move_cmd = Twist()
def on_shutdown():
    rospy.loginfo('Clossing ROS Node')

rospy.init_node('NEAT', anonymous=False)
rospy.on_shutdown(on_shutdown)
r = rospy.Rate(10)

class Turtlebot:
    def __init__(self, cmd_vel_publisher, scan_subscriber, odom_subscriber):
        self.cmd_vel = rospy.Publisher(cmd_vel_publisher, Twist, queue_size=5)
        self.teta = 1.57
        self.laser = [2, 2, 2, 2, 2, 2]
        rospy.Subscriber(scan_subscriber, LaserScan, self.get_laser, queue_size=1)

    def get_laser(self, msg):
        self.laser = msg.ranges
        #rospy.loginfo(self.laser)

    def get_theta(self, msg):
        quaternion = [0, 0, 0, 0]
        quaternion[0] = msg.pose.pose.orientation.x
        quaternion[1] = msg.pose.pose.orientation.y
        quaternion[2] = msg.pose.pose.orientation.z
        quaternion[3] = msg.pose.pose.orientation.w
        euler = euler_from_quaternion(quaternion)
        self.teta = euler[2]
        #rospy.loginfo('Orientacion actual: %s' % np.rad2deg(Teta1))

def eval_genomes(genomes, config):
    nets = []
    ge = []
    turtlebots = []

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        turtlebots.append(Turtlebot('tb3_%s/cmd_vel' % i, 'tb3_%s/scan' % i, 'tb3_%s/odom' % i))
        g.fitness = 0
        ge.append(g)
    
    score = 0
    t0 = time()
    time1 = 0

    while len(turtlebots) > 0 and time1 <= 120:
        time1 = time() - t0

        # Every second each robot alive recieve 1 point reward
        for x, turtlebot in enumerate(turtlebots):
            ge[x].fitness += 1
                 
            # The only entrance its going to be the polar distance to the closest obstacle
            laser_list = turtlebot.laser             
            output = nets[x].activate((min(laser_list), laser_list.index(min(laser_list)))) 
            move_cmd.linear.x  = output[0]
            move_cmd.angular.z = output[1]

            turtlebot.cmd_vel.publish(move_cmd)

            # If the turtlebot crashes it pops from the population
            if min(turtlebot.laser) < 0.15:
                ge[x].fitness -= 10
                turtlebots.pop(x)
                nets.pop(x)
                ge.pop(x)
                rospy.loginfo('The robot %s crashed' % x)

def run(config_file):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, wich is hte top-level object for a NEAT run
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 300)

#Main function
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')
run(config_path)
