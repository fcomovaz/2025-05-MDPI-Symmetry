# [Title to be set]
> In order to reproduce the experiment results please follow the instructions below

Before you start please be aware of some stuff:
* All the information of how the code is running is saved in a log file named 'YYYY-MM-DD.log'. Useful information is printed there.
* File `pipelines.py` is designed to run from terminal. The execution is `python pipelines.py --stage s --epochs e --embedding m` where $s$ is the stage from where you want start, and $e$ is the number of epochs to run
  * if flag `--stage s` is missing, automatically will take the value of 0
  * if flag `--epochs e` is missing, automatically will take the value of 10
  * if flag `--embedding m` is missing, automatically will take the value of 11
* Alternatively, we just created a shell script called `experiment.sh` that will run the pipeline from terminal
  * `./experiment.sh` will run the whole experient (all embeddings from stage 0 for the 1st one).
  * It can be modified to run certain embeddings or stages as desired.
  * To try it, please make sure to do it executable with `chmod +x experiment.sh` and simply run it with `./experiment.sh`

## Requirements/Tools used

|Tool|Version|
|:--:|:--:|
|Python|3.10.15|
|Cuda|12.0.140|
|OS|Ubuntu 24.04.2 LTS x86_64|
|tensorflow|tf_nightly [2.20.0.dev20250418]|
