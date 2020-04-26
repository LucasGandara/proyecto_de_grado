#!/usr/bin/env python   
# -*- coding: utf-8 -*-

import rospy as ROS
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
    
class Show_orientation():
    
    def __init__(self):
        ROS.init_node('path_and_localization', anonymous=False)
        ROS.on_shutdown(self.shutdown)
        self.cmd_vel = ROS.Publisher('cmd_vel', Twist, queue_size=5)
        self.TETA = 0
        self.burger_pose = {'x': 0.0, 'y': 0.0} # Actual POSE and Orientation
        self.burger_orientation = [0, 0, 0, 0]
        # Odometry subscriber
        ROS.Subscriber("/odom", Odometry, self.get_odometry)
        zROS.spin()

    def get_odometry(self, msg):
        """ Read the value from /odom topic"""
        # Position
        self.burger_pose['y'] = msg.pose.pose.position.y
        self.burger_pose['x'] = msg.pose.pose.position.x

        # Orientation
        self.burger_orientation[0] = msg.pose.pose.orientation.x
        self.burger_orientation[1] = msg.pose.pose.orientation.y
        self.burger_orientation[2] = msg.pose.pose.orientation.z
        self.burger_orientation[3] = msg.pose.pose.orientation.w

        euler = euler_from_quaternion(self.burger_orientation)
        self.TETA = euler[2]
        ROS.loginfo('Orientacion actual: %s' % np.rad2deg(self.TETA))
    
    def shutdown(self):
        """ Function to execute on exit """
        self.cmd_vel.publish(Twist()) # Stop the robot before closing
        ROS.loginfo('Closing Node')

    def getKey(self):
        if os.name == 'nt':
            return msvcrt.getch()
    
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
    
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

Show_orientation()