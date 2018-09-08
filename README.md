# Zalo AI Challenge - Landmark Identification

## Required Packages

```bash
$ pip3 install -r requirements.txt
```

## Data

### Data Preparation

Let's assume that the provided dataset is already in *data* folder:
- data/train_val2018.zip
- data/train_val2018.json

After extracting *train_val2018.zip*, we will have *TrainVal* folder containing all images inside *data*.

We run the code to prepare data for training:
- train/validation with the ratio of 92%/8%.
- splitting it into 10 folds in the cross-validation manner.

```bash
$ python3 src/data_prepare.py
```

After finished, all needed files are in *data* folder and ready for training. The *class_dis.txt* file contains distribution over all classes, which is also created inside *data* and later used for weighted loss in training.

### Data Augmentation

- Random rotation
- Central cropping
- Color distortion
- Aspect distortion
- Random cropping
- Random horizontal flipping

## Training

### Model Architectures

- Inception-ResNet
- ResNet-152

Other architectures are also available: ResNet-50, ResNet-101, ResNet-200, Inception-V4, DenseNet-161, NASNet, PNASNet with pre-trained models can be downloaded from [Slim](https://github.com/tensorflow/models/tree/master/research/slim) package. 

### Model Training

First, we need to download pre-trained models of Inception-ResNet and ResNet-152 from [Slim](https://github.com/tensorflow/models/tree/master/research/slim).

```bash
$ chmod 700 download.sh
$ ./download.sh
```

All pre-trained weights will be stored in *pretrained* folder.

To trained a new model with Inception-ResNet architecture:
```bash
$ python3 src/train.py --net "inception_resnet_v2" --checkpoint_exclude_scopes "InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits" --loss_weighted "linear"
```

With ResNet-152:
```bash
$ python3 src/train.py --net "resnet_v2_152" --checkpoint_exclude_scopes "resnet_v2_152/logits" --loss_weighted "linear"
```

```
optional arguments:
  -h, --help                show this help message and exit
  --data_dir                DATA_DIR
                              Path to data folder (default: data)
  --log_dir                 LOG_DIR
                              Path to data folder (default: log)
  --checkpoint_dir          CHECKPOINT_DIR
                              Path to checkpoint folder (default: checkpoints)
  --num_checkpoints         NUM_CHECKPOINTS
                              Number of checkpoints to store (default: 1)
  --num_epochs              NUM_EPOCHS
                              Number of training epochs (default: 50)        
  --num_threads             NUM_THREADS
                              Number of threads for data processing (default: 8)
  --batch_size              BATCH_SIZE
                              Batch Size (default: 16)
  --image_size              IMAGE_SIZE
                              Size of images after pre-processed (default: 299)
  --net                     NETWORK ARCHITECTURE
                              Network architecture to use (default: inception_resnet_v2)
  --learning_rate           LEARNING_RATE
                              Starting learning rate (default: 0.01)       
  --train_mode              TRAINING_MODE
                              Training mode (default: 2)
  --fold                    DATA_FOLD
                              Fold of data to train from (default: -1)   
  --optimizer               OPTIMIZER
                              Weight update rule (default: momentum)    
  --lr_decay_rule           LR_DECAY_RULE
                              Decay rule for learning rate (default: step)    
  --loss_weighted           LOSS_WEIGHTED
                              Cross entropy weighted loss mechanism (default: linear)
  --pretrained_dir          PRETRAINED_DIR
                              Path to pre-trained weights (default: pretrained)
  --trainable_scopes        TRAINABLE_SCOPES
                              Scopes of variables to be fixed (default: None)
  --ignore_mixing_vars      IGNORE_MIXING_VARS
                              When restoring a checkpoint would ignore missing variables (default: False)
  --allow_soft_placement    ALLOW_SOFT_PLACEMENT
                              Allow device soft device placement (default: True)
```

After training, all models will be stored in *checkpoints* folder.

## Final Ensemble Models

- 2 Inception-ResNet and 1 ResNet-152 models trained on 92% and tested on 8% of data.
- 10 Inception-ResNet models trained on 10 folds of cross-validation.

## Prediction

Run *predict.sh* script with an argument which is the path to the data folder containing images. For example, all test images are in *data/test*:

```bash
$ chmod 700 predict.sh
$ ./predict.sh data/test
```

The final predictions will be in *submission.csv* file right under current directory.

**The code are partially borrowed from [Slim](https://github.com/tensorflow/models/tree/master/research/slim) package including model architectures and data augmentation with some modifications.**
