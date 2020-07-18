#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy as ROS
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from tf.transformations import euler_from_quaternion
import tf
from math import sin, cos, atan2, hypot, pi
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import sys

ROS.init_node('position_control')
print 'Iniciando nodo para el control de posición'

def transform_vel(Ei, teta):
    """ Transform the velocity from the global frame to robot frame """
    Rz = np.array([ [cos(teta), -sin(teta), 0],
                    [sin(teta),  cos(teta), 0],
                    [        0,          0, 1]])
    
    Er = np.matmul(Rz, Ei)
    return Er

def constrain(w):
    """ Limits the given (w) angular velocity to Turtlebot3 model burger
        max velocity: 0.22m/s, 6.6rad/s"""
    VEL_LIMIT = 1000 # rad/s
    w = VEL_LIMIT if w >= VEL_LIMIT else w
    w = -VEL_LIMIT if w <= -VEL_LIMIT else w
    return w

def get_odometry(msg):
    """ Read the value from /odom topic"""
    global burger_pose, TETA
    # Position
    burger_pose['x'] = msg.pose.pose.position.x
    burger_pose['y'] = msg.pose.pose.position.y

    # Orientation
    burger_orientation[0] = msg.pose.pose.orientation.x
    burger_orientation[1] = msg.pose.pose.orientation.y
    burger_orientation[2] = msg.pose.pose.orientation.z
    burger_orientation[3] = msg.pose.pose.orientation.w

    euler = euler_from_quaternion(burger_orientation)
    TETA = euler[2]

# Odometry subscriber
ROS.Subscriber("/odom", Odometry, get_odometry)

# cmd_vel publisher
vel_publisher = ROS.Publisher('cmd_vel', Twist, queue_size=10)
rate = ROS.Rate(50) #[Hz]matplotlib.pyplot as plts
vel = Twist()
L = 0.168 # Wheel separation
R = 0.033 # Wheel radius
t = 0.1   # Time base
XTOTAL = [] # Vector for errror x points 
YTOAL = []  # Vector for error y points
POSXTOTAL = []
POSYTOTAL = []
burger_pose = {'x': 0.0, 'y': 0.0} # Actual POSE and Orientation
burger_orientation = [0, 0, 0, 0]
TETA = 0
LASTTETA = 0
Kv = 1 # control gain
iterations = 0
X_references = [0.6, 0.2, 0.4, 0.6, 1]
Y_references = [0.3, 0.6, 1, 1, 1]
TETA_references = [0, 0, 0, 0, 0]

# Graph variables
Totalerrorx = []
Totalerrory = []
Totalwr     = []
Totalwl     = []
# Model Ecuations
for i in range(1):
    sleep(1)
    X_reference = X_references[i]
    Y_reference = Y_references[i]
    errorx = X_reference - burger_pose['x'];
    errory = Y_reference - burger_pose['x'];
    print 'Referencia # ', i
    print 'Referecenia actual x: ', X_reference
    print 'Referencia acutal y: ', Y_reference
    print '---------------------------'
    # Errror 10% separacion de ruedas 
    while hypot(errorx, errory) > 0.0016:
        iterations += 1

        errorx = X_reference - burger_pose['x'];
        errory = Y_reference - burger_pose['y'];
        Totalerrorx.append(errorx)
        Totalerrory.append(errory)

        """ Control block """
        V  = Kv * hypot(errorx, errory)# gain(Kv) * Actual distance 
        Xp = V * cos(TETA) 
        Yp = V * sin(TETA)
        errort = atan2(errory, errorx) # angle to the reference point
        tetap = errort - TETA
        # Inverse kinematic
        A = np.array([ [R * cos(TETA) / 2, R * cos(TETA) / 2],
                    [R * sin(TETA) / 2, R * sin(TETA) / 2],
                    [R/L              ,-R/L] ])
        
        Ainv = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A)) # Ainv = (A' * A)⁻¹ * A'
        
        W = np.matmul(Ainv, np.array([ [Xp],
                                    [Yp],
                                    [tetap] ]))
        
        # Wheel velocities
        wr = constrain(W.item(0))
        wl = constrain(W.item(1))
        Totalwr.append(wr)
        Totalwl.append(wl)

        # Robot velocities
        H = np.matmul(A, np.array([[wr], [wl]]))
        X = H.item(0) 
        Z = H.item(2)

        Hrobot = transform_vel(H, TETA)
        X = Hrobot.item(0)
        Z = Hrobot.item(2)
        vel.linear.x = min(V, 0.1)

    
        if  errort <= -pi/4 or errort > pi/4:
            if Y_reference < 0 and errory:
                errort = -2 * pi + errort
            elif Y_reference >= 0 and burger_pose['y'] > Y_reference:
                errort = 2 * pi + errort
        if LASTTETA > pi - 0.1 and TETA <= 0:
            TETA = 2 * pi + TETA
        elif LASTTETA < -pi + 0.1 and TETA > 0:
            TETA = -2 * pi + TETA
        vel.angular.z = 1 * errort - TETA

        if vel.angular.z > 0:
            vel.angular.z = min(vel.angular.z, 1.5)
        else:
            vel.angular.z = max(vel.angular.z, -1.5)

        LASTTETA = TETA

        POSXTOTAL.append(burger_pose['x'])
        POSYTOTAL.append(burger_pose['y'])

        # Publish the velocity
        vel_publisher.publish(vel)
        rate.sleep()
    
    print 'Posicion a la cual llegó: ', burger_pose['x'], burger_pose['y']

vel.linear.x = 0
vel.angular.z = 0
for i in range(10):
    vel_publisher.publish(vel)

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.linspace(1, iterations, iterations), Totalerrorx)
axes[0].set_title('Errorx')
plt.setp(axes[0], ylabel='Distance [m]')
plt.setp(axes[0], ylim=(0, max(Totalerrorx) + 0.01))
axes[1].plot(np.linspace(1, iterations, iterations), Totalerrory)
axes[1].set_title('ErrorY')
plt.setp(axes[1], xlabel='Sample')
plt.setp(axes[1], ylabel='Distance [m]')
fig1, axes1 = plt.subplots(2,1)
axes1[0].plot(POSXTOTAL, POSYTOTAL)
axes1[0].set_title('puntos X, Y')
plt.setp(axes1[0], xlabel='X [m]')
plt.setp(axes1[0], ylabel='Y [m]')
plt.setp(axes1[0], xlim = (0, max(POSXTOTAL)))
plt.setp(axes1[0], ylim = (0, max(POSYTOTAL)))

plt.show()