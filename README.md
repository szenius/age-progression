# Age Progression
## Installation
This project was developed with `Python 3.6.7`. Please ensure an equivalent version is installed.

As some dependencies can only be installed via an [Anaconda 3](https://www.anaconda.com/distribution/) environment, it is advisable to first install and set up Anaconda 3. 

Thereafter, please create an Anaconda environment by running this in the Anaconda prompt. This should be done in the root directory of the age-progression project. Feel free to choose any name for `<envname>`.
```
conda create --name <envname> --file conda_requirements.txt

# For example:
# conda create --name age-progression --file conda_requirements.txt
# Here you have created an environment named age-progression
```

Once done, activate the environment and install all pip requirements.
```
conda activate <envname>
pip install -r requirements.txt
```

## User Guide
```
python src/train.py <dataset_path> <load_saved> <num_epoch> <neg_ratio> <batch_size> <model_type>

# eg: python src/train.py "./data/" "True" 1000 3 128 "diff"
```
* `dataset_path`: the path to a directory with the raw image files. Must contain empty `before/` and `after/` directories.
* `load_saved`: True if you want to load extracted images from `<dataset_path>/before` and `<dataset_path>/after` directories instead of extracting them again. Otherwise False.
* `num_epoch`: number of training epochs.
* `neg_ratio`: N where 1:N is the ratio of positive to negative training examples.
* `batch_size`: samples in a batch result in a collective update to the gradient.
* `model_type`: either "diff" or "concat". The first runs a model which takes the absolute difference between two encodings; the second runs a model which concatenates the two encodings.