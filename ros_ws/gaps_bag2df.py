import pandas as pd
import rosbag
import yaml


def extract_gaps_params():
    with open("/home/james/.ros/gaps_params.yaml") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    topics = params["/crazyswarm_server/genericLogTopics"]
    assert len(topics) == 1
    logtopic = topics[0]
    vars_key = f"/crazyswarm_server/genericLogTopic_{logtopic}_Variables"
    vars = params[vars_key]
    assert all(v.startswith("gaps.") for v in vars)
    vars = [v[5:] for v in vars]

    bag = rosbag.Bag("/home/james/.ros/gaps.bag")
    records = []
    for topic, msg, t in bag.read_messages():
        if logtopic in topic:
            assert len(msg.values) == len(vars)
            record = {var: val for var, val in zip(vars, msg.values)}
            record["t"] = t.to_sec()
            records.append(record)
    df = pd.DataFrame(records)
    return df


def extract_trajectory():
    bag = rosbag.Bag("/home/james/.ros/gaps.bag")
    msgs = list(bag.read_messages())
    active = [
        i for i, (topic, msg, t) in enumerate(msgs)
        if topic == "/trial" and msg.data == True
        # == True is unnecessary but emphasizes that msg.data is a bool
    ]
    ibegin = active[0]
    iend = active[-1]
    #t0 = msgs[ibegin][-1].to_sec()
    #t1 = msgs[iend][-1].to_sec()
    #print(f"from {t0} to {t1} (duration {t1 - t0} sec)")
    msgs = msgs[ibegin:(iend + 1)]
    records = []
    for topic, msg, t in msgs:
        tsec = t.to_sec()
        if topic == "/fan":
            records.append(dict(t=tsec, fan=msg.data))
        if topic == "/tf":
            for tf in msg.transforms:
                assert tf.header.frame_id == "world"
                child = tf.child_frame_id
                trans = tf.transform.translation
                pos = (trans.x, trans.y, trans.z)
                if child == "target":
                    records.append(dict(t=tsec, target=pos))
                elif child.startswith("cf"):
                    records.append(dict(t=tsec, pos=pos))
    df = pd.DataFrame(records)
    return df


def main():
    df_params = extract_gaps_params()
    df_trajs = extract_trajectory()
    df = pd.concat([df_params, df_trajs], ignore_index=True)
    print(df.columns.tolist())
    df.to_json("gaps_log.json")



if __name__ == "__main__":
    main()
