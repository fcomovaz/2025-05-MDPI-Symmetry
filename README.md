# [Title to be set]
> In order to reproduce the experiment results please follow the instructions below

Before you start please be aware of some stuff:
* ALl the information of how the code is running is saved in a log file named 'YYYY-MM-DD.log'. Useful information is printed there.
* File `run.py` is designed to run from terminal. The execution is `python run.py --stage s --epochs e` where $s$ is the stage from where you want start, and $e$ is the number of epochs to run
  * if flag `--stage s` is missing, automatically will take the value of 0
  * if flag `--epochs e` is missing, automatically will take the value of 10
* File `pipeline.py` is designed to run without flags, the stage can be modified inside the code, just run `python pipeline.py`

## Requirements/Tools used

|Tool|Version|
|:--:|:--:|
|Python|3.10.15|
|Cuda|12.0.140|
|OS|Ubuntu 24.04.2 LTS x86_64|
|tensorflow|tf_nightly [2.20.0.dev20250418]|
