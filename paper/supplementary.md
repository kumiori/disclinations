# Supplementary code for the paper: Title of paper
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/..../demo.ipynb)

This repository contains supplementary code for the paper
> Finsberg, H., Dokken, J. 2022.
> Title of paper, Journal of ..., volume, page, url


## Abstract
Overview

## Getting started

We provide a pre-build Docker image which can be used to run the the code in this repository. 
First, ensure that you have [docker installed](https://docs.docker.com/get-docker/).

To start an interactive docker container you can execute the following command

```bash
docker run --rm -ti -v "$(pwd)":/home/numerix -w /home/numerix kumiori3/numerix:stable
```

## Data

Data is available in a dropbox folder. Use the script `download_data.sh` in the data folder to download the data.

The data folder should have the following structure after the data is downloaded.
```
├── README.md
├── data.tar
├── download_data.sh
└── mesh
    ├── heart01.msh
    └── heart02.msh
```
These meshes are originally taken from <https://>, but since the original data is about `size` we decided to make a smaller dataset for this example.

Eventually when we publish a paper we could put this data on e.g [Zenodo](https://zenodo.org). That will make sure the data gets it's own DOI.


## Scripts
All the scripts are located in the folder called `xxx` in the repository. Is is assumed that we run the script from within this folder.

### Pre-processing
In order to reproduce the results we need to first run the pre-processing script
```
python3 pre_processing.py
```
This will `...`

### Run
The next step is to run the code. We do this by running the script
```
python3 zxy.py
```
This will create a new folder `code/output` containing files called `xxx`.

### Postprocessing
The final step is to postprocess the results by running the script
```
python3 postprocess.py
```
This will generate a file for visualizing the `...` in the Paraview (inside `code/results` called  `...xdmf`). This script will also compare some features. If the results differ, then the program will raise an error.


## Citation

```
@software{Our_Research_Software_2024,
  author = {Foo, Bar and Baz, Hew},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{Our Research Software}},
  url = {https://github.com/disclinations},
  version = {0.0.1},
  year = {2024}
}
```


## Having issues
If you have any troubles please file and issue in the GitHub repository.

## License
GPLv3