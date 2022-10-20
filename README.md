# JEMGL

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)



This package contains the JEMGL algorithm for solving joint network topology inference problems. <br>

## Getting started

### Install via pip

The package is available on pip and can be installed with

    pip install JEMGL

### Install from source

Alternatively, you can install the package from source using the following commands:

    git clone https://github.com/yanli4360/JEMGL-master.git
    pip install -r requirements.txt
    python setup.py

Test your installation with 

    pytest JEMGL/ -v


### Advanced options

When installing from source, you can also install dependencies with `conda` via the command

	$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

If you wish to install `JEMGL` in developer mode, i.e. not having to reinstall `JEMGL` everytime the source code changes (either by remote or local changes), run

    python setup.py clean --all develop clean --all



