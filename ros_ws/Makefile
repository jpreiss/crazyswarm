ROS:=$(HOME)/.ros

fan_params.pdf: \
$(ROS)/fan_default.json \
$(ROS)/fan_gaps.json
	python3 gaps_analyze.py fan $^

weight_params.pdf: \
$(ROS)/weight_default.json \
$(ROS)/weight_gaps.json
	python3 gaps_analyze.py weight $^

bad_init_params.pdf: \
$(ROS)/diag_bad_init_gaps.json \
$(ROS)/diag_bad_init_nogaps.json \
$(ROS)/diag_good_init.json
	python3 gaps_analyze.py bad_init $^

$(ROS)/%.json: gaps_bag2df.py $(ROS)/%_params.yaml $(ROS)/%_config.json $(ROS)/%.bag
	python3 $< $*
