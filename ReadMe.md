# Chexoplanet 

A python modelling code for exoplanets observed by CHEOPS.

This code natively:
* Downloads CHEOPS data for a given target using `dace-query`
* Automatically processes CHEOPS data using PSF photometry
* Downloads TESS, K2, Kepler & CoRoT data for a target using `MonoTools.lightcurve`, and natively includes them in any transit fitting
* Automatically takes TOI information to initialise transit fits of known candidates/planets
* Automatically accesses survey data to provide precise stellar parameters
* Assesses linear and quadratic decorrelation parameters for individual and multiple CHEOPS visits in order to leverage extra data for improved lightcurve performance.
* Enables GP or spline as a function of rollangle for high-frequency roll angle variations.
* Performs Bayesian model comparison between (CHEOPS detrending + planet) and (CHEOPS detrending only) models to assess the reliability of transit detections
* Allows direct incorporation of RV timeseries via `exoplanet`
* Allows TTV modelling via `exoplanet`
* Performs Hamiltonian Monte Carlo sampling using PyMC3, including in highly correlated parameter spaces (using PyMC3_ext)
* Produces a wide variety of plots and tables for publication 
* Is written in a pythonic manner which enables in-notebook manipulation rather than text-file-only inputs.

## General Installation instructions
* Pip install all the packages we need using `pip install numpy pandas pymc3-ext exoplanet celerite2 astropy astroquery h5py httplib2 lxml tess-cloud scipy patsy pycheops urllib3 seaborn iteround everest-pipeline`
* In order to install the new dace query module (which would break exoplanet if lumpy-1.23 is installed), run: “pip install --no-deps dace-query”. [You may need to run the following command to make sure all dependencies installed: pip install "astropy<6.0.0" "pyerfa>=2.0" "PyYAML>=3.13" "packaging>=19.0" "python-dateutil>=2.8.1" "pytz>=2020.1" "charset-normalizer<4" "idna<4" "urllib3<3" "certifi>=2017.4.17" "six>=1.5”]
* And now download the important projects from git, into e.g. a `/python` folder: `git clone http://github.com/hposborn/MonoTools`, `git clone http://github.com/hposborn/chexoplanet`, and `git clone http://github.com/alphapsa/PIPE`
* cd into PIPE.
* Download the PIPE config files (e.g. bias map, darks, flats, etc) from Alexis Brandeker's dropbox  - he can email the link
* Modify the config files to tell it where these config files are found by editing `PIPE/pipe/config/conf.json` and make "ref_lib_data”:"/location/of/PIPE/config/data”.
* For the PIPE data_root folder, choose somewhere useful like "/home/USERNAME/home1/python/PIPE/pipe/data”
* Now pip install pipe by running `pip install .` in the `home1/python/PIPE` directory (and while on your conda env)
* Generate a DACE authentication key which puts a dace password into a .dacerc file (see: https://dace.unige.ch/tutorials/?tutorialId=26) You may need to request access to get CHEOPS GTO data.

## Installation steps specific to unibe horus cluster:
* Create new conda environment using `conda create --name chexo python=3.8` (python-3.8 is slightly softer on older libraries)
* Launch the conda environment with  `conda activate chexo`
* Check the python and pip installs are pointing to the correct subfolder using `which pip` and `which python`
* You may need to conda install pip using `conda install pip`
* Download anaconda distribution from https://www.anaconda.com/products/distribution#linux
* install anaconda in /home/USERNAME/home1 using `bash Anaconda3-2022.05-Linux-x86_64.sh` [don’t put it in /home/USERNAME]
* Modify the .bashrc to look in /home/USERNAME/home1 by adding `export XDG_CONFIG_HOME=/home/USERNAME/home1` to the .bashrc
* Create a local python folder `mkdir /home/USERNAME/home1/python`
* Add it to your python path `export PATH=/home/USERNAME/home1/python:$PATH` (this tells python to look here when importing modules)
* Move into this directory `cd /home/USERNAME/home1/python`
* For the PIPE ref_lib_data you can directly use `/shares/home1/hosborn/data/PIPE_data``.
* Copy (or directly link using `ln`) your `.dacerc` file to `cd /home/USERNAME/home1/`