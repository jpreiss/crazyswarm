#!/usr/bin/python3

import glob
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
    devs = glob.glob("/dev/ttyACM*")
    if len(devs) == 0:
        raise IOError("No Arduino serial device found.")
    dev = devs[0]
    print("using device " + dev)
    arduino = serial.Serial(dev,  baudrate=115200, timeout=1)
    # turn off initially
    arduino.write(bytes("0", "ascii"))
    rospy.init_node("fan_switcher")
    rospy.Subscriber("fan", std_msgs.msg.Bool, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
