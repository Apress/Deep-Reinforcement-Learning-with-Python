# drl-2ed

Source Code for the book "Deep Reinforcement Learning with Python", second edition by Nimish Sanghi

### Local Install - Ubuntu and Windows WSL2
1. Please install following ubuntu packages using:
```
apt-get install swig cmake ffmpeg freeglut3-dev xvfb git-lfs
git lfs install
```
2. Create a new `venv` or `conda` environment with `python=3.9`. Activate it and then inside the environment, install packages from `requirements.txt` using following command:
```
pip install -r requirements.txt
```

3. clone the repository in a local drive. Navigate the the folder where it is cloned and start jupyterlab using command `

### Local Install - macOS

1. Please install following packages using `brew`:
```
brew install swig cmake ffmpeg freeglut3 git-lfs
git lfs install
```

2. Create a new `venv` or `conda` environment with `python=3.9`. Activate it and then inside the environment, install packages from `requirements.txt` using following command:
```
pip install --use-pep517 pymunk
pip install -r requirements.txt
```

3. clone the repository in a local drive. Navigate the the folder where it is cloned and start jupyterlab using command `


### Running on Colab
1. Open the notebook on Google Colab either by uploading from local drive or by directly connecting Google Colab with github. Each notebook is self contained and ready to be run Colab or local.
2. To run on Colab, you will need to uncomment and execute the code cells under section with heading "Running in Colab/Kaggle". 
3. Unless specified, you can run the code in regular CPU environments in Colab.

#### Refer to Chapter 1 of the book for a step by step process of setting up environment on the platform of your choice. You may also refer to the pdf "Installation_steps.pdf" in this repository.