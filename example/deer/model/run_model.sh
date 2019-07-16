#!/usr/bin/env bash
set -e

# Source directory of this repository
paper_source=$PAPER_2019_PHYSICAL_DATA_MODEL

# Directory of CMake build files for this repository
paper_objects=$PAPER_2019_PHYSICAL_DATA_MODEL_OBJECTS

export PATH="$paper_objects/lue/bin:$PATH"

model_prefix="$paper_source/example/deer/model"
lue=$paper_source/lue

# Location where all generated files will be written to
output_prefix=/tmp/deer

lue_dataset=$output_prefix/deer.lue
dot_properties=$lue/document/lue_translate/dot_properties.json
dot_graph=$output_prefix/deer.dot
pdf_graph=$output_prefix/deer.pdf

mkdir -p $output_prefix

# Run model
$model_prefix/model.py --nr_timesteps=250 --nr_deer=25 $lue_dataset

# Create graph with structure of dataset
lue_translate export --meta $dot_properties $lue_dataset $dot_graph
dot -Tpdf -o $pdf_graph $dot_graph

ls --human-readable --color=auto -l $output_prefix
