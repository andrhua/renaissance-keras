# Renaissance

Simple keras model for drawings recognition set up to be uploaded on Google Cloud Platform for faster learning.

### Prerequisites

Clone this repo:

```
git clone https://github.com/andrhua/renaissance-keras.git
```

Download (~50 GB) and preprocess [dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) (see next section).

For cloud training, install and setup [Google SDK](https://cloud.google.com/sdk/).

For local training, install Python >3.4 and [Tensorflow](https://www.tensorflow.org/install).

### Preproccess data

First, you need to convert `.npy` arrays to `.tfrecords` for further creating [tf.Dataset](https://www.tensorflow.org/guide/datasets).

Reasons to DO NOT use `.npy` arrays directly:
* A decent dataset with huge number of examples will not fit in RAM even on Google Cloud machines, at least on free-tier plan :)
* Given a classification problem, we should shuffle our dataset, eliminating [selection bias](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias); in other words, you will never touch some part of data for learning, reducing your model's ability to generalize. Shuffling numpy arrays requires to keep them in memory.
* It is a good practice and has a lot of other advantages in pipelining training process.
On average, every drawing has ~150k examples, but for ~85% top-5 accuracy 10k examples per class is enough.

```
create_tfrecords \
--src=path/to/numpy/arrays \ 
--train_dst=path/to/write/train/tfrecords \
--eval_dst=path/to/write/eval/tfrecords
--train_size=10000
--eval_size=2000
```
`.tfrecords` takes more space than numpy arrays, so all processed data could take >100 GB apart from numpy arrays themselves.

If you are going to train model locally, go to the next section.
If you are going to train model in Google Cloud, you are need to upload tfrecords to bucket on Google Storage, and also grant access to this bucket to your project.

### Launch a training job
#### On local machine
```
# see optional training-specific arguments in train.py
python trainer/train.py \
--train_src=path/to/train/tfrecords
--eval_src=path/to/eval/tfrecords
```

#### On Google Cloud platform
Carefully read official [guide](https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs) first.

Then open training configuration file `config.yaml` in any text editor and edit `jobDir` and `region` properties according to your project location.

Add shell variables:
```
TRAINER_PACKAGE_PATH="/path/to/cloned/repo"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="your_name_$now"
MAIN_TRAINER_MODULE="trainer.task"
PACKAGE_STAGING_PATH="gs://your/chosen/staging/path"
```
Finally, submit a training job:
```
gcloud ml-engine jobs submit training $JOB_NAME \
        --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --config config.yaml
```
After training completion you can grab output directory with Tensorflow model and deploy it in any way you like: export to Tensorflow Lite, [upload to a Google Cloud](https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models) for online predictions and so on.

## Acknowledgments

* [Base keras model](https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)
