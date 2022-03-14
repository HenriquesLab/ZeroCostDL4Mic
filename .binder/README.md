# Running on Jupyter

| Notebook   | Jupyter | Colab |   |   |
|------------|---------|-------|---|---|
| StarDist2D | &#9745; | &#9745; |   |   |
| Example    | &#9744; | &#9745; |   |   |
|            |         |       |   |   |


## Binder

## Binder local (repo2docker)

Install [repo2docker](https://github.com/jupyterhub/repo2docker), using conda, pip 

### No GPU

    repo2docker .

### (nVidia) GPU

Ensure nvidia-docker is install on machine https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html:

Check with:

    docker run --gpus all --privileged nvidia/cuda:11.4.0-runtime-ubuntu20.04 nvidia-smi

To build ZeroCost locally with a tag run:

    repo2docker --no-run  --image-name zerocostdl4mic .

To build the repo locally with a tag
    
    docker run --publish=8888:8888 --gpus all --privileged zerocostdl4mic

Then follow the token link

## Conda (local)

Ensure anaconda/miniconda/mamba are installed and run

    conda env create -f .binder/environment.yml --force

From the root dir of the repo.

    conda activate dl4mic

To run a notebook (e.g. StarDist2D)

    jupyter

or 
    jupyter lab



<!-- A basic binderised repository with GPU support -->
