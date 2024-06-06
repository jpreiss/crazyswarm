from pathlib import Path
import json
import sys

import pandas as pd
import rosbag
import yaml


def convert(parampath, configpath, bagpath, outpath):
    # Parse the launch file so we can give proper names to the anonymous values
    # in the CF generic log block.
    with open(parampath) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    with open(configpath) as f:
        config = json.load(f)
    topics = params["/crazyswarm_server/genericLogTopics"]
    assert len(topics) == 1
    logtopic = topics[0]
    vars_key = f"/crazyswarm_server/genericLogTopic_{logtopic}_Variables"
    vars = params[vars_key]
    assert all(v.startswith("gaps6DOF.") for v in vars)
    vars = [v[9:] for v in vars]

    try:
        kind = config["optimizer"]
    except KeyError:
        # old format, before multiple optimizers
        kind = "gaps" if config["gaps"] else "none"
    if kind == "none":
        kind = "detune" if config["detune"] else "baseline"

    # Use the "trial" topic to isolate the part where we measure performance.
    bag = rosbag.Bag(bagpath)
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
    df["optimizer"] = kind
    for k, v in config.items():
        df[k] = v
    df.to_json(outpath)


if __name__ == "__main__":
    root = Path.home() / ".ros"
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
    else:
        bags = prefix.glob("*.bag")
        newest = max(bags, key=lambda p: p.stat().st_mtime)
        prefix = newest.stem
    print("loading:", prefix)
    convert(
        root / f"{prefix}_params.yaml",
        root / f"{prefix}_config.json",
        root / f"{prefix}.bag",
        root / f"{prefix}.json",
    )