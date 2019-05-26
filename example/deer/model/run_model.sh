#!/usr/bin/env bash
set -e

source=$PAPER_2019_PHYSICAL_DATA_MODEL
objects=$PAPER_2019_PHYSICAL_DATA_MODEL_OBJECTS

export PATH="$objects/lue/bin:$PATH"

model_prefix="$source/example/deer/model"
lue=$source/lue
output_prefix=/tmp/deer

lue_dataset=$output_prefix/deer.lue
dot_properties=$lue/document/lue_translate/dot_properties.json
dot_graph=$output_prefix/deer.dot
pdf_graph=$output_prefix/deer.pdf

mkdir -p $output_prefix
$model_prefix/model.py $lue_dataset
lue_translate export --meta $dot_properties $lue_dataset $dot_graph
dot -Tpdf -o $pdf_graph $dot_graph
$model_prefix/visualize.py $lue_dataset --output $output_prefix
ls --human-readable --color=auto -l $output_prefix
