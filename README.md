# Model-based Offline Multi-Agent Dialogue Policy Learning

This repository reflects the final version for my Reinforcement Learning based internship at [Saarthi.ai](https://saarthi.ai). The intermediate commits have been removed for privacy reasons.

The complete research and code implementation was based on the paper "Multi-Agent Task-Oriented Dialog Policy Learning with Role-Aware Reward Decomposition", Arxiv Link at [arxiv](https://arxiv.org/abs/2004.03809) and "Is Your Goal-Oriented Dialog Model Performing Really Well? Empirical Analysis of System-wise Evaluation", Arxiv Link at [arxiv](https://arxiv.org/abs/2005.07362).



## Codes to run for SGDS Data

### SGDS Data

Firstly, Download the SGDS Data and run the code `datagen.py`. Make sure the SGDS-DSTC Data is already present in folder named `dstc8-schema-guided-dialogue-master`.

`datagen.py` is a comprehensive data generator which generates major files in the folder `dstc-data` required as pre-requisite for pre-training.

#### Pre-Training

Secondly, Pre-Train the model to generate datasets.

```
python3 main.py --pretrain True --save_dir model_pre --data_dir dstc-data --config dstcsgds
```

Point to be noted, if,  dialog policy sys and usr don't run simultaenously, re-run the above code.

#### Training

Sampling of data is done using the multiprocessing, but due to some issues, it's not working properly. Function: `sampler_dstc`, file: `learner.py`

```
python3 main.py --load model_pre/best --lr_policy 1e-4 --save_dir model_RL --save_per_epoch 4 --data_dir dstc-data --config dstcsgds
```




### Codes to run for Original Data

Adapted from Ryuichi Takanobu's [Github](https://github.com/truthless11/MADPL)

unzip [zip](https://drive.google.com/open?id=1S2RXrXwsajrdzyyvM0ca_BLfGdb0PBgD) under `data` directory, or simply running

```
mkdir data/
cd data
gdown https://drive.google.com/uc?id=1S2RXrXwsajrdzyyvM0ca_BLfGdb0PBgD
unzip MADPLdata.zip
```

`gdown` can be installed via pip: `pip install gdown`


the pre-processed data are under `data/processed_data` directory

- data preprocessing will be automatically done if `processed_data` directory does not exists when running `main.py`

#### Use with Original Data

the best trained model is under `data/model_madpl` directory

```
python main.py --test True --load data/model_madpl/selected > result.txt
```

#### Run

Command

```
python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

*distributed RL* is implemented for parallel trajectory sampling. You can set `--process` to change the number of multi-process, and set `--batchsz_traj` to change the number of trajectories each process collects before one update iteration.

#### pretrain

```
python main.py --pretrain True --save_dir model_pre
```

**NOTE**: please pretrain the model first

#### train

```
python main.py --load model_pre/best --lr_policy 1e-4 --save_dir model_RL --save_per_epoch 1
```

#### test

```
python main.py --test True --load model_RL/best
```