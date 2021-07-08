# SMM detection in dynamic feature space using wearable sensors

This repository contains the code for 'Stereotypical Motor Movement Detection in Dynamic Feature Space', by Nastaran Mohammadian Rad, Seyed Mostafa Kia, Calogero Zarbo, Giuseppe Jurman, Paola Venuti, and Cesare Furlanello.

If you use the code in this repository for a published research project, please cite this paper.

The code is designed to run on Python 2.7 and Keras 1.0.8.

# Replication of results

1. Preprocessing_Simulated_Data.py
* Input: sourceDataFilesAcc, sourceDataFilesGyro, sourceDataFilesMag. The path of accelerometer, gyroscope, and magnetometer data for three sessions of one subject. Change the input to the path where you save data. Example: ~/Simulated_Data/subject path/session path/EXLAccelerometer.csv 
* Output: Preprocessed data (saved in preprocessed folder for each session)
After preprocessing, all data are saved in output files: preprocessedDataFilesAcc, preprocessedDataFilesGyro, preprocessedDataFilesMag i.e., the path of preprocessed accelerometer, gyroscope, and magnetometer data for three sessions of one subject. Example: ~/Simulated_Data/subject path/ session path/ EXLAccelerometer.csv
* Run Preprocessing_Simulated_Data.py to generate preprocessed data: EXLAccelerometer.csv, EXLGyroscope.csv, EXLMagnetometer.csv files. To express data in the standard measurement units each direction of each IMU sensor modality i.e., accelerometer, gyroscope, and magnetometer signals are multiplied by predefined values (i.e., accCoeff, gyrCoeff, magCoeff). Furthermore, the band-pass filter with a cut-off frequency of 0.1 Hz is applied to remove DC components in signal.   


2. segmentation_Simulated_Data.py 
* Input: Set the input path (Acc1_preprocessed, Acc2_preprocessed, …, Mag3_preprocessed) to the path of preprocessed data per each session of each subject. Furthermore, to produce labels, set the data1_log, data2_log, and data3_log  to the path of annotation files, i.e., ~/subject path/session path/events_AndroidDevice-Device/LogSensor.csv.
* Output: 
   * annotation_subjectX: labels before segmentation for each subject.
   * concatenated_IMU_subject: concatenated data (accelerometer, gyroscope, and magnetometer before segmentation).
   * SubjectX: feature matrix and corresponding labels after segmentation.
   *  All outputs are saved in the Intermediate_files folder.

3. CNN_Simualetd_Data.py: 
* Input: Feature matrix and corresponding labels for each subject. Change the path to the intermediate_files path. To replicate results of the Static-Features-Balanced experiment, set balanced variable to 1. To replicate results of the Static-Features-Unbalanced experiment change the balanced variable to 0. 
* Output: Set the output directory to the path of CNN_balanced and CNN_Unbalanced in the Result folder. Each folder contains:
   * “Simulated_CNN_Results.mat” including evaluation metrics.
   * CNN_learned_features_sub_x_run_ including learned features for each run and each subject to be used as input for the LSTM layer. 
   * Learned model and weights.
4. LSTM_Simulated_Data: 
* Inputs: Set the input path to the CNN_Unbalanced folder path, i.e., learned features by CNN on unbalanced data are used as input for LSTM layer.
* Outputs: 
   * “Simulated_CNN_LSTM_Prediction_Sub_X” LSTM results on different time steps and neuron numbers. 
    * Model and weights.