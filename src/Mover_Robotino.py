#!/usr/bin/env python

import sys
import rospy
import time
from gazebo_msgs.msg import ModelState

class Combination():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.flag = False
        self.type =int(sys.argv[1]) 
        self.turtle = ModelState()
        self.turtle.model_name = 'turtlebot3_burger'
        self.turtle.pose.position.x = 0
        self.turtle.pose.position.y = 0
        self.turtle.pose.orientation.x = 0
        self.turtle.pose.orientation.y = 0
        self.turtle.pose.orientation.z = 0
        self.turtle.pose.orientation.w = 0

        self.obs1 = ModelState() 
        self.obs2 = ModelState()
        self.obs3 = ModelState()
        self.obs1.model_name = 'unit_box_0'
        self.obs2.model_name = 'unit_box_1'
        self.obs3.model_name = 'unit_box_2'
        for i in range(4):
            self.pub_model.publish(self.turtle)
            time.sleep(0.1)

        self.moving()

    def moving(self):
        if self.type == 0:
            self.tipo0()

        if self.type == 1:
            self.tipo1()

        if self.type == 2:
            self.obs1.pose.position.x = 3.544239
            self.obs1.pose.position.y = -4.899666
            self.obs2.pose.position.x = 5
            self.obs2.pose.position.y = 5
            self.obs3.pose.position.x = 8
            self.obs3.pose.position.y = 9
            for i in range(4):
                self.pub_model.publish(self.obs1)
                time.sleep(0.1)
                self.pub_model.publish(self.obs2)
                time.sleep(0.1)
                self.pub_model.publish(self.obs3)
                time.sleep(0.1)
            time.sleep(2.9)
            self.tipo2(self.obs1)

        if self.type == 3:
            self.obs1.pose.position.x = 2.744239
            self.obs1.pose.position.y = -4.899666
            self.obs2.pose.position.x = 4.121827
            self.obs2.pose.position.y = -1.762737
            self.obs3.pose.position.x = 5
            self.obs3.pose.position.y = -5 
            for i in range(4):
                self.pub_model.publish(self.obs1)
                time.sleep(0.1)
                self.pub_model.publish(self.obs2)
                time.sleep(0.1)
                self.pub_model.publish(self.obs3)
                time.sleep(0.1)
            self.tipo3(self.obs1)

    def tipo0(self):
        self.obs1.pose.position.x = 3.082989
        self.obs1.pose.position.y = 0 
        self.obs2.pose.position.x = 4.0832
        self.obs2.pose.position.y = 0
        self.obs3.pose.position.x = 4.432605 
        self.obs3.pose.position.y = -1.2284
        for i in range(4):
            self.pub_model.publish(self.obs1)
            time.sleep(0.1)
            self.pub_model.publish(self.obs2)
            time.sleep(0.1)
            self.pub_model.publish(self.obs3)
            time.sleep(0.1)

    def tipo1(self):
        self.obs1.pose.position.x = 4.064512
        self.obs1.pose.position.y = -2.431163
        self.obs2.pose.position.x = 1.493949
        self.obs2.pose.position.y = -2.502498
        self.obs3.pose.position.x = 5
        self.obs3.pose.position.y = 5
        for i in range(4):
            self.pub_model.publish(self.obs1)
            time.sleep(0.1)
            self.pub_model.publish(self.obs2)
            time.sleep(0.1)
            self.pub_model.publish(self.obs3)
            time.sleep(0.1)

    def tipo2(self, model):
        while not self.flag:
            model.pose.position.y += 0.007

            if model.pose.position.y >= -0.5:
                self.flag = True

            self.pub_model.publish(model)
            time.sleep(0.1)

    def tipo3(self, model):
        vely = 0.009
        velx = 0
        while not self.flag:
            model.pose.position.y += vely
            model.pose.position.x += velx
            if model.pose.position.y >= -1.782:
                vely = 0
                velx = -0.02

            if model.pose.position.x <= 1.4:
                self.obs3.pose.position.x = 1.38396
                self.obs3.pose.position.y = -2.946
                self.pub_model.publish(self.obs3)
                time.sleep(0.1)
                self.flag = True

            self.pub_model.publish(model)
            time.sleep(0.1)

def main():
    rospy.init_node('MoveObstacle')
    rospy.loginfo('Move obstacle node Initialized')
    try:
        combination = Combination()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
