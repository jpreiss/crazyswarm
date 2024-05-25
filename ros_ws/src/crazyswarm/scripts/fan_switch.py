#!/usr/bin/python3

import glob
import serial
import time

import rospy
import std_msgs


arduino = None
CHR_L = ord("l")
READY_MSG = bytes("Arduino is ready.", "ascii")


def fan_callback(msg):
    global arduino
    s = "f1" if msg.data else "f0"
    arduino.write(bytes(s, "ascii"))


def light_callback(msg):
    global arduino
    rgb = [int(255 * c) & 0xFF for c in [msg.r, msg.g, msg.b]]
    w = 0
    b = bytes([CHR_L, w] + rgb)
    arduino.write(b)


def main():
    global arduino
    devs = glob.glob("/dev/ttyACM*")
    if len(devs) == 0:
        raise IOError("No Arduino serial device found.")
    dev = devs[0]
    print("using device " + dev)
    arduino = serial.Serial(dev, baudrate=115200, timeout=1)

    # wait for arduino to be ready
    while True:
        line = arduino.readline()
        # because it sends weird line endings
        if line.startswith(READY_MSG):
            break
    print("arduino is ready.")

    # turn fan off initially
    arduino.write(bytes("f0", "ascii"))
    # turn light off initially
    b = bytes([CHR_L, 0, 0, 0, 0])
    arduino.write(b)

    rospy.init_node("fan_switcher")
    rospy.Subscriber("fan", std_msgs.msg.Bool, fan_callback, queue_size=1)
    rospy.Subscriber("light", std_msgs.msg.ColorRGBA, light_callback, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
