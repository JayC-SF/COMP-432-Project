# COMP-432-Project

This project uses MiniConda for the python environment as it is simpler to deal with package management especially regarding legacy GPUs.

Please find the MiniConda user guide to install it on your computer.
Please also find the environment.yml file to see all the libraries installed for this project.

If you do not have Miniconda installed, please find this link for setting it up.

If you alread have Miniconda installed, make sure to update to the latest version as follows:
```sh
conda update -n base -c defaults conda
```


To create the conda environment for this project use the following command:
```sh
conda env create -f environment.yml -n <new_env_name>
```

