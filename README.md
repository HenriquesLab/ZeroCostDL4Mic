# dl4mic

Packaged form of [ZeroCostDl4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) to make the process more platform agnostic.
Attempts to bundle reusable code and structure model training and prediction into a no-code config file toolset.


    pip install git+https://github.com/ctr26/dl4mic


Currently working with Noise2Void and Care2D

## Build and test

This project uses poetry to build, test and manage dependnecies:

Quick start:
    peotry build
    poetry install
    poetry run pytest

Note that testing is (rightly) slow due to running model epochs for testing

## Todo:

- Find all the bugs
- Implement the full roster of ZeroCostDL4Mic models.
- Command line interface
- Implement lazy loading of large/uninstalled packages (looking at you pyTorch)
