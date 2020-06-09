# D-H
D-H assignment - image processing

This repository contains my solution to the Docler-Holding AI team's take home assignment.

The task was to classify the cifar-10 dataset.

The data can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html. I was unable to upload it here, as it is too large. 
"modified_mobilenet_1k_baseline.py" contains the source code. The model was trained in Google Colab with the notebook "modified_mobilenet_train.ipynb".

#### "Report.ipynb" contains the findigs and the analysis of the problem.

Run "run_experiment.py" to reproduce the result. The trained model's weights are "modified_model_1k_2" and this can be downloaded from https://www.dropbox.com/s/w7nbcmbmygqdo4c/modified_model_1k_2?dl=0 . (File is too big to ulpoad)
For this the following have to installed:
`pip3 intall torch torchvision` `pip3 install matplotlib` `pip3 install numpy`
