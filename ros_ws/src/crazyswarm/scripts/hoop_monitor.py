#!/usr/bin/python3

import numpy as np
import rospy
import std_msgs
from tf2_msgs.msg import TFMessage


GOOD = 0.05 # 8 cm
BAD = 0.5
CENTER_YZ = np.array([-0.65, 0.99])

pub = rospy.Publisher("light", std_msgs.msg.ColorRGBA, queue_size=1)
msg = std_msgs.msg.ColorRGBA()
was_bad = False


def callback(tfmsg):
    global was_bad
    for tf in tfmsg.transforms:
        if tf.header.frame_id != "world":
            continue
        if not tf.child_frame_id.startswith("cf"):
            continue
        p = tf.transform.translation
        msg.r = 0
        msg.g = 0
        if abs(p.x) < 0.2 and p.y < 0:
            if abs(p.x) < 0.1:
                brightness = 1
            else:
                brightness = 1.0 - (abs(p.x) - 0.1) / 0.1
            xz = np.array([p.y, p.z])
            dist = np.linalg.norm(xz - CENTER_YZ)
            if dist < GOOD and not was_bad:
                msg.r = 0
                msg.g = brightness
            elif dist < BAD or was_bad:
                was_bad = True
                msg.r = brightness
                msg.g = 0
        else:
            was_bad = False  # reset
        pub.publish(msg)


def main():
    rospy.init_node("hoop_monitor")
    rospy.Subscriber("tf", TFMessage, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
