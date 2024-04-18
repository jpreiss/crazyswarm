#!/usr/bin/python3

import serial
import time

import rospy
import std_msgs


arduino = None


def callback(msg):
    global arduino
    s = "1" if msg.data else "0"
    arduino.write(bytes(s, "ascii"))


def main():
    global arduino
    arduino = serial.Serial("/dev/ttyACM1",  baudrate=115200, timeout=1)
    # turn off initially
    arduino.write(bytes("0", "ascii"))
    rospy.init_node("fan_switcher")
    rospy.Subscriber("fan", std_msgs.msg.Bool, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
