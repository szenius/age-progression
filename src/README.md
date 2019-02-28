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
python src/train.py "./data/testset/" "True"
```