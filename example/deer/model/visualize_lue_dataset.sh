set -e

# Source directory of this repository
paper_source=$PAPER_2019_PHYSICAL_DATA_MODEL

# Blender's python does not contain the packages we need to convert lue
# datasets to visualizations: lue, numpy, ... Given that our Python
# version equals the one shipped with Blender, we can point Blender's
# Python to our Python's site-packages.
python_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`

export PYTHONPATH="$PYTHONPATH:$paper_source/source:$python_site_packages"

# Currently, this script assumes the data is stored in /tmp/deer
script=visualize_lue_dataset.py

# Start Blender, create visualization, present interface
/opt/blender/blender --python $script
