# 2019_physical_data_model
This repository contains a version of the LUE physical data model as
presented in our 2019 manuscript, as well as example scripts and other
files used in the preparation of that manuscript.

| directory | contents |
| --------- | -------- |
| `example` | Deer-biomass model referred to from manuscript |
| `lue` | Version of LUE described in manuscript |
| `source` | Scripts used for visualising example-model output |

The most recent LUE source code can be found in LUE's [own
repository](https://github.com/pcraster/lue).

![Deer tracks](deer_tracks.png)


## Build LUE Python package
LUE is currently developed and tested on Linux using GCC-7. All code
should compile and run fine on other platforms too, but this is not
regularly tested.

Here is an example session of building the version of LUE used for our
manuscript and installing the LUE targets in `$HOME/lue_install`:

```bash
cd /tmp
# Recursive is used to also checkout submodules
git clone --recursive https://github.com/pcraster/paper_2019_physical_data_model
cd paper_2019_physical_data_model
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/lue_install ..
cmake --build . --target install
```

The LUE data model source code depends on 3rd party libraries and tools,
that may or may not be installed already on your system. The following
dependencies can usually be installed using your system's package manager.

| package | version used |
| ------- | ------------ |
| libboost-dev | 1.65.1 |
| libgdal-dev | 2.2.3 |
| hdf5-dev | 1.10.0 |

These package names correspond with those used in Debian distributions
and derivatives. Other versions of these packages might also work.

Unless provided by your system's package manager also, these prerequisites
can be installed using [Conan](https://conan.io/):

| package | version used |
| ------- | ------------ |
| fmt | 5.2.1 |
| gsl_microsoft | 2.0.0 |
| jsonformoderncpp | 3.5.0 |
| pybind11 | 2.2.4 |

Other versions of these packages might also work.

To install Conan, some additional Python packages, and the above
prerequisites, [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
(or Conda) can be used:

```bash
conda env create -n test \
    -f ../lue/environment/configuration/conda_environment.yml
conda activate test
conan install ../conanfile.txt
```

Once LUE is installed, some commandline utilities can be found in
`$HOME/lue_install/bin` and the Python package in
`$HOME/lue_install/python`.


# Use LUE Python package
To be able to use the LUE commandline utilities and Python package,
the following environment variables must be set as follows:

```bash
export PATH=$PATH:$HOME/lue_install/bin
export PYTHONPATH=$PYTHONPATH:$HOME/lue_install/python
```

Now these commands should not result in errors:

```bash
lue_validate
python -c "import lue"
```

Here is an example session of using the LUE Python package. An empty
dataset is created and validated.

Python script:
```python
# create_dataset.py
import lue

dataset = lue.create_dataset("my_first_dataset.lue")
```

Shell commands:
```bash
python create_dataset.py
lue_validate my_first_dataset.lue
```


# Run example model
The following commands can be used to run the example model referred to
from the manuscript:

```bash
../example/deer/model/model.py --nr_timesteps=250 --nr_deer=25 deer.lue
```

To visualize the model output, [Blender](https://www.blender.org) can
be used. For information how this has been done when preparing the
manuscript, see `run_model.sh` and `visualize_lue_dataset.sh` in the
example model directory.
