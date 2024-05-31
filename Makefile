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

bad_init_params.pdf: \
gaps_analyze.py \
$(ROS)/diag_bad_init_gaps.json \
$(ROS)/diag_bad_init_nogaps.json \
$(ROS)/diag_bad_init_singlepoint.json \
$(ROS)/diag_bad_init_ogd.json \
$(ROS)/diag_good_init.json
	python3 $^ bad_init

$(ROS)/%.json: gaps_bag2df.py $(ROS)/%_params.yaml $(ROS)/%_config.json $(ROS)/%.bag
	python3 $< $*
