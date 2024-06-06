ROS:=$(HOME)/.ros

fan_params.pdf: \
gaps_analyze.py \
$(ROS)/fan_default.json \
$(ROS)/fan_gaps.json
	python3 $^ fan

weight_params.pdf: \
gaps_analyze.py \
$(ROS)/weight_default.json \
$(ROS)/weight_gaps.json
	python3 $^ weight

# TODO: figure out the right way to make these repetitive deps

bad_init_params.pdf: \
gaps_analyze.py \
$(ROS)/diag_bad_init_gaps.json \
$(ROS)/diag_bad_init_nogaps.json \
$(ROS)/diag_bad_init_singlepoint.json \
$(ROS)/diag_bad_init_episodic.json \
$(ROS)/diag_bad_init_ogd.json \
$(ROS)/diag_good_init.json
	python3 $^ bad_init

episodic.pdf: \
gaps_analyze.py \
$(ROS)/episodic_gaps.json \
$(ROS)/episodic_500.json \
$(ROS)/episodic_750.json \
$(ROS)/episodic_875.json \
$(ROS)/episodic_1000.json \
$(ROS)/episodic_1250.json \
$(ROS)/episodic_1500.json \
$(ROS)/episodic_1750.json \
$(ROS)/episodic_2000.json \
$(ROS)/episodic_2250.json \
$(ROS)/episodic_2500.json \
$(ROS)/episodic_3000.json
	python3 $^ episodic

$(ROS)/%.json: gaps_bag2df.py $(ROS)/%_params.yaml $(ROS)/%_config.json $(ROS)/%.bag
	python3 $< $*
