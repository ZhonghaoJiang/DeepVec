# DeepVec: State-Vector Aware Test Case Selection for Enhancing Recurrent Neural Network

DeepVec has conducted experiments on the combination of 6 models and data sets. To illustrate the usage of the code of this project, the example of testing LSTM model trained with MNIST data set is illustrated as follows.

## Environment

python=3.7

```sh
pip install -r requirements.txt
```
Please place the SVHN training set and test set under `./data/` and `./RNNModels/svhn_demo/data/`.

[//]: # (## Preparing an RNN model as the test object)

[//]: # ()
[//]: # (```sh)

[//]: # (cd RNNModels/mnist_demo)

[//]: # (python3 mnist_lstm.py -type "train")

[//]: # (```)

[//]: # ()
[//]: # (After the training is completed, the output is as follows, and the trained model will be saved in the `./RNNModels/mnist_demo/models/mnist_lstm.h5`.)

[//]: # ()
[//]: # (```)

[//]: # (Epoch 20/20)

[//]: # (54000/54000 [==============================] - 10s 188us/step - loss: 0.0112 - accuracy: 0.9963 - val_loss: 0.0548 - val_accuracy: 0.9878)

[//]: # (```)

[//]: # ()
[//]: # (## Preparing the data set for selection)

[//]: # ()
[//]: # (For evaluating RQ1, we generate 30 different dataset for selection:)

[//]: # ()
[//]: # (```sh)

[//]: # (# generate the augmented data for selection)

[//]: # (cd ../../gen_data/gen_test_dataset)

[//]: # (python3 dau_mnist.py)

[//]: # (python3 gen_toselect_dataset.py -dataset "mnist"   # for RQ1 & RQ2)

[//]: # (```)

[//]: # ()
[//]: # (For evaluation RQ3, we generate the dataset for selection and retraining and the test set for evaluation:)

[//]: # ()
[//]: # (```sh)

[//]: # (# generate the augmented data for selection and retraining)

[//]: # (cd ../gen_retrain_dataset)

[//]: # (python3 dau_mnist.py)

[//]: # (python3 gen_retrain.py -dataset "mnist"   # for RQ3)

[//]: # (```)

[//]: # ()
[//]: # (## Generating the abstract model used for calculating the DeepStellar-coverage )

[//]: # ()
[//]: # (The coverage calculation of DeepStellar requires an abstract model to be generated in advance. This part of the code comes from [DeepStellar]&#40;https://github.com/xiaoningdu/deepstellar&#41; 's open source code.)

[//]: # ()
[//]: # (```sh)

[//]: # (cd ../..)

[//]: # (python3 ./abstraction_runner.py -test_obj "mnist_lstm")

[//]: # (```)

## We provide a pipeline that does all the preliminary preparation
```sh
bash prepare.sh
```


## RQ1: Bug Detection Rate
```sh
bash RQ1_run.sh
```

The results will be saved in `./exp_results/rq1` .

## RQ2: Inclusiveness

```sh
bash RQ2_run.sh
```

The results will be saved in `./exp_results/rq2` .

If you need to evaluate the inclusiveness on the original test set and the augmented test set independently, execute the following code:
```sh
bash RQ2_seperate.sh
```


## RQ3: Fault Diversity

```sh
bash RQ3_run.sh
```

The results will be saved in `./exp_results/rq3` .

## RQ4: Guidance

Run the following code to get the accuracy of the retrained model:

```sh
bash RQ4_run.sh
```

If you need to retrain with all candidate sets to get the benchmark value, execute the following code

```sh
bash RQ4_100_run.sh
```

The results will be saved in `./exp_results/rq4` .

## RQ5: Time Cost

```sh
bash RQ5_run.sh
```

The results will be saved in `./exp_results/rq5` .

## RQ6: Ablation Study
```sh
bash ablation.sh
```

The results will be saved in `./exp_results/ablation` .

## RQ7: Sensitive Analysis

```sh
bash Sensitive.sh
```

The results will be saved in `./exp_results/sensitive` .
