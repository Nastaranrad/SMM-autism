"""
@author: Nastaran Mohammadian Rad, <nastaran@fbk.eu>
Paper title: "Stereotypical Motor Movement Detection in Dynamic Feature Space"
"""
import pandas as pd
from scipy.signal import butter, lfilter

#-----------------get data for one subject---------------------------
# put the path of three sessions of accelerometer sensors 
sourceDataFilesAcc = ['~/Simulated_Data/Subject 01/Raw data/01/data_EXLs3Device-IMU 1/EXLAccelerometer.csv',
         '~/Simulated_Data/Subject 01/Raw data/02/data_EXLs3Device-IMU 1/EXLAccelerometer.csv',
         '~/Simulated_Data/Subject 01/Raw data/03/data_EXLs3Device-IMU 1/EXLAccelerometer.csv']
# put the path of three sessions of gyroscope sensors for one subject 
        
sourceDataFilesGyro = ['~/Simulated_Data/Subject 01/Raw data/01/data_EXLs3Device-IMU 1/EXLGyroscope.csv',
         '~/Simulated_Data/Subject 01/Raw data/02/data_EXLs3Device-IMU 1/EXLGyroscope.csv',
         '~/Simulated_Data/Subject 01/Raw data/03/data_EXLs3Device-IMU 1/EXLGyroscope.csv']
# put the path of three sessions of magnetometer sensors 
         
sourceDataFilesMag= ['~/Simulated_Data/Subject 01/Raw data/01/data_EXLs3Device-IMU 1/EXLMagnetometer.csv',
         '~/Simulated_Data/Subject 01/Raw data/02/data_EXLs3Device-IMU 1/EXLMagnetometer.csv',
         '~/Simulated_Data/Subject 01/Raw data/03/data_EXLs3Device-IMU 1/EXLMagnetometer.csv']
         
#----------set the output path to save the preprocessed data-----------------
# set the path for saving the preprocessed data for three sessions of accelerometer
preprocessedDataFilesAcc=['~/Simulated_Data/Subject 01/Preprocessed_data/01/data_EXLs3Device-IMU 1/EXLAccelerometer.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/02/data_EXLs3Device-IMU 1/EXLAccelerometer.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/03/data_EXLs3Device-IMU 1/EXLAccelerometer.csv']
# set the path for saving the preprocessed data for three sessions of gyroscope

preprocessedDataFilesGyro=['~/Simulated_Data/Subject 01/Preprocessed_data/01/data_EXLs3Device-IMU 1/EXLGyroscope.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/02/data_EXLs3Device-IMU 1/EXLGyroscope.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/03/data_EXLs3Device-IMU 1/EXLGyroscope.csv']
# set the path for saving the preprocessed data for three sessions of magnetometer
         
preprocessedDataFilesMag=['~/Simulated_Data/Subject 01/Preprocessed_data/01/data_EXLs3Device-IMU 1/EXLMagnetometer.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/02/data_EXLs3Device-IMU 1/EXLMagnetometer.csv',
         '~/Simulated_Data/Subject 01/Preprocessed_data/03/data_EXLs3Device-IMU 1/EXLMagnetometer.csv']

#--------Functions definition-------------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

#------------set parameters-----------------------------------
FSAMP = 100
Low = 0.1 
High = 45 
FONDOSCALA_ACC = 16
FONDOSCALA_GYR = 2000

accCoeff = FONDOSCALA_ACC*4/32768.0
gyrCoeff = FONDOSCALA_GYR/32768.0
magCoeff = 0.007629

#---Apply butter_bandpass_filter on IMU signals
#save the results
for f in range(len(sourceDataFilesAcc)):
    dataFrame=pd.read_csv(sourceDataFilesAcc[f])
    dataFrame['ax'] = butter_bandpass_filter(dataFrame['ax']*accCoeff, Low, High, FSAMP)
    dataFrame['ay'] = butter_bandpass_filter(dataFrame['ay']*accCoeff, Low, High, FSAMP)       
    dataFrame['az'] = butter_bandpass_filter(dataFrame['az']*accCoeff, Low, High, FSAMP)
    dataFrame.to_csv(preprocessedDataFilesAcc[f], index= False) 


for f in range(len(sourceDataFilesGyro)):
    dataFrame=pd.read_csv(sourceDataFilesGyro[f])
    dataFrame['gx'] = butter_bandpass_filter(dataFrame['gx']*gyrCoeff, Low, High, FSAMP)
    dataFrame['gy'] = butter_bandpass_filter(dataFrame['gy']*gyrCoeff, Low, High, FSAMP)       
    dataFrame['gz'] = butter_bandpass_filter(dataFrame['gz']*gyrCoeff, Low, High, FSAMP)
    dataFrame.to_csv(preprocessedDataFilesGyro[f], index= False) 

    
for f in range(len(sourceDataFilesMag)):
    dataFrame=pd.read_csv(sourceDataFilesMag[f])    
    dataFrame['mx'] = dataFrame['mx']*magCoeff
    dataFrame['my'] = dataFrame['my']*magCoeff     
    dataFrame['mz'] = dataFrame['mz']*magCoeff
    dataFrame.to_csv(preprocessedDataFilesMag[f], index= False) 
    

    



