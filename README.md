# Sleep-stages-recognition
Recognition sleep stages based on PSG (polysomnography) signals using machine learning method. The reason of creating this project was my future Master degree thesis.

## Idea

The idea of this work is an attempt of creating such automated system that can classify a sleep stage based on a part of the PSG signal. As an input dataset for training should contain different signals such as EEG, EMG, EOG and also hypnogram with sleep stages. The hypnogram can help model to understand in which part of the signals we have for example NREM or REM stages. On the image below, you can see example of the hypnogram

<p align = "center">
  <img src ="/images/hypnogram.png" width="800" >
</p>

As the result I should get a model  based on one of the various neural networks. Putting for example a 30 seconds duration signals into this model, on the output we can get a sleep stage (for example NREM, REM, N1, etc.). So in the end we can preprocess a whole PSG data during one night and get a predicted hypnogram.

There are two folders in this repositorium: nn_images and nn_signals. The first one contains program for predicting sleep stage based on the image of the one chosen signal (for example eeg). The second folder contains program for predicting sleep stages based on the numerical signals.


# Folder nn_images
### Creating model

As it was mentioned earlier the proper type of Neural Network for the goal of this project is LSTM. But I found out that also possible can be CNN type which will be created the first. 

The model will be created in Python language. 

### Creating dataset

In the first step we should create a python file for creating dataset for our CNN. Dataset include different stages images, that are stages Wake, NREM (1, 2, 3, 4) and REM. These stages will be provided by different signals SC from the [*Sleep-EDF Database Expanded*](https://www.physionet.org/content/sleep-edfx/1.0.0/) PhysioNet dataset. This file is called **create_dataset.ipynb** which was created in Jupytr Lab.

Description of the functions:

- file_keeper(psg_file, hypnogram_file) - get as the input two files: PSG and Hypnogram. As the output function gives raw data and raw annotations.
- all_sleep_stages(annotations) - get as the annotations. As the output function gives all sleep stages of the PSG signal file.


