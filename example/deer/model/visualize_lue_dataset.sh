set -e

# Blender's python does not contain the packages we need to convert lue
# datasets to visualizations: lue, numpy, ... Given that our Python
# version equals the one shipped with Blender, we can point Blender's
# Python to our Python's site-packages.
python_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`

export PYTHONPATH="$PYTHONPATH:$PAPER_2019_PHYSICAL_DATA_MODEL/source:$python_site_packages"

script=visualize_lue_dataset.py

/opt/blender/blender --python $script
# /opt/blender/blender --background --python $script
# gdb --args /opt/blender/blender --background --python $script
