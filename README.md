# Voice Conversion Project

This project intends to convert any to any voice conversion

## Set up

Create your own virtual environment. Anaconda v3 is used for this project.
```bash
conda create -n jglvc python=3.9
```
Then, install the following packages after activating the environment:
```bash
conda activate jglvc
conda install -c pytorch pytorch=1.10
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge librosa
conda install scikit-learn
conda install -c metric-learning pytorch-metric-learning
pip install torchsummary
```
