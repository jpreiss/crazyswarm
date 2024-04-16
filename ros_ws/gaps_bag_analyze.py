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
    fan_times = [t for topic, _, t in msgs]


def main():
    df_params = extract_gaps_params()
    print(df_params)



if __name__ == "__main__":
    main()
