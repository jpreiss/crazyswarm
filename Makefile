SHELL := /bin/bash

ROS := ./data/final

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

compare_params.pdf: \
gaps_analyze.py \
$(ROS)/fan_gaps.json \
$(ROS)/weight_gaps.json
	python3 $^ multi_param

BAD_INIT := $(shell echo $(ROS)/bad_init_{gaps,detuned,modelfree,episodic,episodicstar,expert}_{1,2,3,4,5}.json)

bad_init_params.pdf: gaps_analyze.py $(BAD_INIT)
	python3 $^ bad_init
# $(ROS)/diag_bad_init_ogd.json \

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
