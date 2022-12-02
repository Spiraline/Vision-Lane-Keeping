#!/usr/bin/env python3
import rospy
from autoware_msgs.msg import VehicleCmd
from geometry_msgs.msg import TwistStamped

def twist_cb(data):
    msg = VehicleCmd()
    msg.twist_cmd = data
    vehicle_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('vehicle_cmd_publihser')

    vehicle_pub = rospy.Publisher("vehicle_cmd", VehicleCmd, queue_size = 10)

    rospy.Subscriber("/twist_cmd", TwistStamped, twist_cb)

    rospy.spin()