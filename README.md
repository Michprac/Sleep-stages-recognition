# Sleep-stages-recognition
Recognition sleep stages based on PSG (polysomnography) signals using machine learning method. The reason of creating this project was my future Master degree thesis.

## Idea

The whole idea of this work is an attempt of creating such automated system that can classify a sleep stage based on a part of the PSG signal.

As the result I should get a model  based on one of the various neural networks (probably it would be stack of the LSTM). Putting for example a 30 seconds duration signal into this model, on the output we can get a sleep stage (for example NREM, REM, N1, etc.). So in the end we can preprocess a whole PSG data during one night and get a hypnogram.
