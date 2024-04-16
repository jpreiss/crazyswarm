import pandas as pd
import rosbag
import yaml


def main():
    # Parse the launch file so we can give proper names to the anonymous values
    # in the CF generic log block.
    with open("/home/james/.ros/gaps_params.yaml") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    topics = params["/crazyswarm_server/genericLogTopics"]
    assert len(topics) == 1
    logtopic = topics[0]
    vars_key = f"/crazyswarm_server/genericLogTopic_{logtopic}_Variables"
    vars = params[vars_key]
    assert all(v.startswith("gaps.") for v in vars)
    vars = [v[5:] for v in vars]

    # Use the "trial" topic to isolate the part where we measure performance.
    bag = rosbag.Bag("/home/james/.ros/gaps.bag")
    msgs = list(bag.read_messages())
    active = [
        i for i, (topic, msg, t) in enumerate(msgs)
        if topic == "/trial" and msg.data == True
        # == True is unnecessary but emphasizes that msg.data is a bool
    ]
    ibegin = active[0]
    iend = active[-1]
    msgs = msgs[ibegin:(iend + 1)]

    # Process the messages.
    records = []
    for topic, msg, t in msgs:
        tsec = t.to_sec()
        if logtopic in topic:
            assert len(msg.values) == len(vars)
            record = {var: val for var, val in zip(vars, msg.values)}
            record["t"] = tsec
            records.append(record)
        elif topic == "/fan":
            records.append(dict(t=tsec, fan=msg.data))
        elif topic == "/tf":
            for tf in msg.transforms:
                assert tf.header.frame_id == "world"
                child = tf.child_frame_id
                trans = tf.transform.translation
                pos = (trans.x, trans.y, trans.z)
                if child == "target":
                    records.append(dict(t=tsec, target_x=trans.x, target_y=trans.y, target_z=trans.z))
                elif child.startswith("cf"):
                    records.append(dict(t=tsec, pos_x=trans.x, pos_y=trans.y, pos_z=trans.z))
    df = pd.DataFrame(records)
    df = df.groupby("t").first().reset_index()
    df.to_json("gaps_log.json")


if __name__ == "__main__":
    main()
