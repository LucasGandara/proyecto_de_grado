#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy as ROS
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from math import atan2, pi, sin, cos
import time
from tqdm import tqdm
""" Turtlebot 3 Burger constants """
R = 0.033 # [m]
L = 0.16  # [m]

vr = 0.2 #[rad/s]
vl = 0.2#[rad/s]

""" Velocity Node Publisher """
ROS.init_node('move_robot')
print 'Iniciando nodo para las pruebas de velocidad'

vel_publisher = ROS.Publisher('/cmd_vel', Twist, queue_size=10)
rate = ROS.Rate(50) #[Hz]
vel = Twist()

start = time.time()
end = time.time()

""" Odometry Subscriber """
burger_pose = {'x':0.0, 'y':0.0}

def get_odometry(msg):
    burger_pose['x'] = msg.pose.pose.position.x
    burger_pose['y'] = msg.pose.pose.position.y

ROS.Subscriber('/odom', Odometry, get_odometry)
ROS.loginfo('Initialiting odom_listener topic') 

tqdm_bar = tqdm(total=20)
while (end - start) <= 20:
    #Calculate linear velocity
    teta = atan2(burger_pose['y'], burger_pose['x']) 
    vel.linear.x = 0.15
    vel.angular.z = 0.0

    vel_publisher.publish(vel)
    rate.sleep()
    end = time.time()
    tqdm_bar.update(end - start)

# tell the car to stop after 10 seconds
vel.linear.x = 0.0
vel.linear.y = 0.0
vel.angular.z = 0.0
vel_publisher.publish(vel)

print 'Cerrando nodo prueba de velocidades'