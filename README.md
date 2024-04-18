# Sleep-stages-recognition
Recognition sleep stages based on PSG (polysomnography) signals using machine learning method. The reason of creating this project was my future Master degree thesis.

## Idea

The idea of this work is an attempt of creating such automated system that can classify a sleep stage based on a part of the PSG signal. As an input dataset for training should contain different signals such as EEG, EMG, EOG and also hypnogram with sleep stages. The hypnogram can help model to understand in which part of the signals we have for example NREM or REM stages.

As the result I should get a model  based on one of the various neural networks (probably it would be stack of the LSTM). Putting for example a 30 seconds duration signal into this model, on the output we can get a sleep stage (for example NREM, REM, N1, etc.). So in the end we can preprocess a whole PSG data during one night and get a hypnogram.

## Creating model

The model will be created in Python language.
