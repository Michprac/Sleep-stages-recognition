# Sleep-stages-recognition
Recognition of the sleep stages based on PSG (polysomnography) signals using machine learning methods. The reason of creating this project was my future Master degree thesis.

## Idea

The idea of this work is an attempt of creating such automated system that can classify a sleep stage based on a part of the PSG signal. As an input dataset for training should contain different signals such as EEG, EMG, EOG and also hypnogram with sleep stages. The hypnogram can help model to understand in which part of the signals we have for example NREM or REM stages. On the image below, you can see example of the hypnogram

<p align = "center">
  <img src ="/images/hypnogram.png" width="800" >
</p>

As the result I should get a model  based on one of the various neural networks. Putting for example a 30 seconds duration signals into this model, on the output we can get a sleep stage (for example NREM, REM, N1, etc.). So in the end we can preprocess a whole PSG (Polysomnography) data during one night and get a predicted hypnogram.

There are two folders in this repositorium, which represent two libraries: nn_images and nn_signals. The first one contains functions for predicting sleep stage based on the images of the one chosen signal (in this case eeg Fpz-Cz). The second folder contains functions for predicting sleep stages based on the extracted features from the PSG channels (in this case 4 channels - eeg Fpz-Cz, eeg Pz-Oz, eog, emg).

Both libraries weere written in Python programming language using Jupyter Lab. The [*Sleep-EDF Database Expanded*](https://www.physionet.org/content/sleep-edfx/1.0.0/) archive was used for training process. In this work *sleep-cassette* files were used.

# Library nn_images

The purpose of this library is to check CNN model, based on images of 30-seconds segments of EEG signal. So the first channel of the PSG data in the *.edf* files was splitted into 30-seconds segments, from which images were created datasets: train data(*sleep-cassette*), test data (*x_test_y_test*), valid data (*x_valid_y_valid*). The directory tree of the library you can find below. Also notice, that after downloading archive with *.edf* files, you should manually split it in appropriate folders and paste them instead of *PASTE_HERE_EDF_FILES.txt* files. 

Appropriate images then go as an input for the CNN model. Training process starts. Structure of the CNN model you can see on the picture below.

<p align = "center">
  <img src ="/images/cnn_structure.png" width="800" >
</p>

After the model was trained, user can predict sleep stages for files in the *x_test_y_test* directory. As the result you can get next files (in this case for the candidate SC4762):

Comparison of the two hypnograms *cnn_hypnogram_SC4762.png*. Such image helps to visualize similarity of the predictions and hypnogram created by a specialist. As you can see there are a lot of noises in this CNN method, so not ideal for using in the real world cases:

<p align = "center">
  <img src ="/images/cnn_hypnogram_SC4762.png" width="800" >
</p>

For the numerical analysis, user gets *.txt* file *cnn_info_txt_eeg_channel.png* with percentage correct detection for the each stage:

<p align = "center">
  <img src ="/images/cnn_info_txt_eeg_channel.png" width="800" >
</p>

Also useful characteristic for describing the result can be confusion map *cnn_confusion_map_SC4762.png*. It helps to see how much stages were detected correctly:

<p align = "center">
  <img src ="/images/cnn_confusion_map_SC4762.png" width="800" >
</p>

Library functions descriptions:

- **cnn_predict.ipynb** – use for predicting, creates confusion maps, *.txt* files and hypnograms
- **cnn_prepare_dataset.ipynb** – use for creating test, validation, training and predicting datsets. Dataset include different stages images: Wake, NREM (1, 2, 3, 4) and REM
- **cnn_train.ipynb** – creates CNN model and evaluate its' accuracy
- **exit_code.ipynb** – additional file for entire execution code. Warning! Jupyter Notebooks at this moment can be important using *import* word, so a single file with all necessary functions was created, use it
- **single_main_file.ipynb** – a single file with all necessary functions was created, USE IT



# Library nn_signals

As you can see, results for CNN mdodel are not acceptable, so I decided to check another metho. The purpose of this library is to check MLP model, based on extracted features from 4 channels of PSG *sleep-cassette* signals. Features are extractd from 30-seconds segments of such signals as  eeg Fpz-Cz, eeg Pz-Oz, eog and emg. Some functions were created to extract appropriate feature for the each electrophysiological signal. Some of these features are mean value, K complex ratio, skewness, eye blink frequency, etc. Extracted feautres are stored not in folders as it was in the previous situation, but in variables (for example for the training data it looks like *features_train* for numerical values and *labels_train* for stage classes). The directory tree of the library you can find below. Also notice, that after downloading archive with *.edf* files, you should manually split it in appropriate folders and paste them instead of *PASTE_HERE_EDF_FILES.txt* files. 

Created features with classes then go as an input for the MLP model. Training process starts. Structure of the MLP model you can see on the picture below.

<p align = "center">
  <img src ="/images/mlp_structure.png" width="800" >
</p>

After the model was trained, user can predict sleep stages for files in the *x_test_y_test* directory. As the result you can get next files (in this case for the candidate SC4761E):

Comparison of the two hypnograms *mlp_all_channels_hypnogram_SC4761E.png*. As you can see there are some noises ther, but it's much better than in the previous situation:

<p align = "center">
  <img src ="/images/mlp_all_channels_hypnogram_SC4761E.png" width="800" >
</p>

For the numerical analysis, user gets *.txt* file *mlp_info_txt_all_channels_SC4761E.png* with percentage correct detection for the each stage:

<p align = "center">
  <img src ="/images/mlp_info_txt_all_channels_SC4761E.png" width="800" >
</p>

Confusion map *mlp_all_channels_confusion_map.png*. It helps to see how much stages were detected correctly:

<p align = "center">
  <img src ="/images/mlp_all_channels_confusion_map_SC4761E.png" width="800" >
</p>

Library functions descriptions:

- **feature_extraction_process.ipynb** – use for extraction features from 4 channels (eeg Fpz-Cz, eeg Pz-Oz, eog and emg)
- **predicting_dataset.ipynb** – use for creating prediction files
- **preparing_dataset.ipynb** – prepare datasets for training process
- **training_process.ipynb** – use for MLP model initialization and training
- **exit_code.ipynb** – additional file for entire execution code. Warning! Jupyter Notebooks at this moment can be important using *import* word, so a single file with all necessary functions was created, use it
- **single_main_file.ipynb** – a single file with all necessary functions was created, USE IT


# Cloud service for machine learning purpose

In case of the CNN model, training process can take a lot of time. So Cloud computig was used here. I used Machine Learning Studio provided by Microsoft Azure. In this service user can rent some compute and storage and train model remotely. User have to create workspace to start working with notebooks, compute and training process. 

<p align = "center">
  <img src ="/images/studio_azure_ml_example.png" width="800" >
</p>

In the studio, a computing virtual machine "Standard_F16s_v2" was used with the following characteristics:
- Number of cores: 16
- Size of temporary operating memory (RAM): 32 GB
- Size of permanent memory: 128 GB
- Type of computing unit: CPU

# Summary

In the summary I can say that the MLP network method is much more better than CNN. The effectiveness of the CNN model is low and amounts to approximately 32.81%. The effectiveness of the MLP model is high and amounts to approximately 75.56%. However, it is still not so accurate enough to completely replace a specialist.

Also some tests have been done on how different combinations of PSG channels affect the accuracy of the MLP model. The networks were trained on the features extracted for these different cases and compared on the bar chart *channels_comparison.png*. The best accuracy value I have got for the next signals combination: eeg Fpz-Cz, eeg Pz-Oz, eog.

<p align = "center">
  <img src ="/images/channels_comparison.png" width="800" >
</p>

# Future improvements

In the future some functions in this library can be modified. For example, remove double generating of the test dataset.

Also a key element that can be changed is architectures of the NN models. In this case, I have used very simple elemenets, like Dense layer, CNN layer.

As an additional feature for this project, GUI can be created. Graphical interface can help to control different activities of the training process.

# Repositorium tree

```bash
│   README.md
│
├───images
│       ...
│
├───nn_images
│   │   cnn_predict.ipynb
│   │   cnn_prepare_dataset.ipynb
│   │   cnn_train.ipynb
│   │   exit_code.ipynb
│   │   single_main_file.ipynb
│   │
│   └───edf_files
│       ├───sleep-cassette
│       │       PASTE_HERE_EDF_FILES.txt
│       │
│       ├───x_test_y_test
│       │       PASTE_HERE_EDF_FILES.txt
│       │
│       └───x_valid_y_valid
│               PASTE_HERE_EDF_FILES.txt
│
└───nn_signals
    │   exit_code.ipynb
    │   feature_extraction_process.ipynb
    │   predicting_dataset.ipynb
    │   preparing_dataset.ipynb
    │   single_main_file.ipynb
    │   training_process.ipynb
    │
    └───edf_files
        ├───sleep-cassette
        │       PASTE_HERE_EDF_FILES.txt
        │
        ├───x_test_y_test
        │       PASTE_HERE_EDF_FILES.txt
        │
        └───x_valid_y_valid
                PASTE_HERE_EDF_FILES.txt
```

# Bibliography

B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen and J. J. L. Oberye, "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG," in IEEE Transactions on Biomedical Engineering, vol. 47, no. 9, pp. 1185-1194, Sept. 2000, doi: [10.1109/10.867928](https://ieeexplore.ieee.org/document/867928).
keywords: {Feedback loop;Electroencephalography;Neurofeedback;Hospitals;Biomedical measurements;Power measurement;Frequency measurement;Maximum likelihood detection;Maximum likelihood estimation;Skull},

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).

Stanislas Chambon, Mathieu N. Galtier, Pierrick J. Arnal, Gilles Wainrib, and Alexandre Gramfort. A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(4):758–769, 2018. doi:[10.1109/TNSRE.2018.2813138](https://ieeexplore.ieee.org/document/8307462).





