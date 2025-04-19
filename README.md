# DeepVec: State-Vector Aware Test Case Selection for Enhancing Recurrent Neural Network


## Environment Setup

```sh
conda create -n deepvec python=3.7
pip install -r requirements.txt
```
Please place the SVHN training set and test set under `./data/` and `./RNNModels/svhn_demo/data/`.

DeepVec depends on NLTK data, please following the guidance to download dependency data when code raise error.
We provide a pipeline that does all the preliminary preparation
```sh
bash prepare.sh
```


## How to run?
To reproduce the experiment reaults, please run the scripts commands in `./scripts`. 
### RQ1: Bug Detection Rate
```sh
bash RQ1_run.sh
```

The results will be saved in `./exp_results/rq1` .

### RQ2: Inclusiveness

```sh
bash RQ2_run.sh
```

The results will be saved in `./exp_results/rq2` .

If you need to evaluate the inclusiveness on the original test set and the augmented test set independently, execute the following code:
```sh
bash RQ2_seperate.sh
```


### RQ3: Fault Diversity

```sh
bash RQ3_run.sh
```

The results will be saved in `./exp_results/rq3` .

### RQ4: Guidance

Run the following code to get the accuracy of the retrained model:

```sh
bash RQ4_run.sh
```

If you need to retrain with all candidate sets to get the benchmark value, execute the following code

```sh
bash RQ4_100_run.sh
```

The results will be saved in `./exp_results/rq4` .



### RQ5: Time Cost

```sh
bash RQ5_run.sh
```

The results will be saved in `./exp_results/rq5` .



### RQ6: Ablation Study
```sh
bash ablation.sh
```

The results will be saved in `./exp_results/ablation` .



### RQ7: Sensitive Analysis

```sh
bash Sensitive.sh
```

The results will be saved in `./exp_results/sensitive` .

## Acknowledgements

- [DeepState](https://github.com/SSCT-Lab/DeepState)
- [ATS](https://github.com/SSCT-Lab/ATS)
- [RTS](https://github.com/swf1996120/RTS)
- [DRFuzz](https://github.com/youhanmo/DRFuzz)