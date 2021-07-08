
"""
@author: Nastaran Mohammadian Rad <Email: nastaran@fbk.eu>
Paper title: "Stereotypical Motor Movement Detection in Dynamic Feature Space"
------------------------------------------------------
This code is to segment and label the preprocessed data. 
Input: annotations (logSensor.csv files) and preprocessed data for each subject.
Output: feature matrix and corresponding labels.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

#---Variable definition---------
subject = 1 # change betwen 1 to 5
samplingFreq = 100
overlap = 10 

# reading annotation file "LogSensor.csv" for 3 sessions of one subject.
data1_log=pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Raw data/01/event_AndroidDevice/LogSensor.csv') #session1 of subject 1
data2_log=pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Raw data/02/event_AndroidDevice/LogSensor.csv') #session 2 of subject 1
data3_log=pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Raw data/03/event_AndroidDevice/LogSensor.csv') #session 3 of subject1

# reading preprocessed data for one subject------------
# three sessions of accelerometer
Acc1_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/01/EXLAccelerometer.csv') #acc session1 sub1
Acc2_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/02/EXLAccelerometer.csv') # acc session2 sub1
Acc3_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/03/EXLAccelerometer.csv') # acc session3 sub1
#three sessions of gyroscope
Gyro1_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/01/EXLGyroscope.csv') # gyro session1 sub1 
Gyro2_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/02/EXLGyroscope.csv') # gyro session 2 sub1
Gyro3_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/03/EXLGyroscope.csv') # gyro session 3 sub1 
#three sessions of magnetometer
Mag1_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/01/EXLMagnetometer.csv') # magno session1 sub 1
Mag2_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/02/EXLMagnetometer.csv') # magno session2 sub1 
Mag3_preprocessed= pd.read_csv('~/smm-detection/Simulated_Data/Subject 01/Preprocessed_data/03/EXLMagnetometer.csv') # magno session3 sub1


savePath = '~/smm_detection/Results/Intermediate_files/'
# label for each session.
label1= np.zeros(Acc1_preprocessed.shape[0],int)
label2= np.zeros(Acc2_preprocessed.shape[0],int)
label3= np.zeros(Acc3_preprocessed.shape[0],int)

#-----function definitions---------

def log_matrix(data_log):
    # this function prepares the label matrix for each session (before segmentation)
    logdata=data_log['timestamp'].as_matrix()
    logdata[:,]=np.round((logdata[:,]-logdata[0,])/10**6)
    logdata = logdata[6:len(logdata)-1,] #removing the last raw
    return logdata

def sensor_sessions(sensor_preprocessed):
    #this function prepares the accelerometer, gyroscope, and magnetometer data for each session
    Acc_matrix= sensor_preprocessed.as_matrix() 
    Acc_matrix[:,0]=np.round((Acc_matrix[:,0]-Acc_matrix[0,0])/10**6)
    return Acc_matrix
    
logdata1 = log_matrix(data1_log)
logdata2 = log_matrix(data2_log)
logdata3 = log_matrix(data3_log)

Acc1_matrix = sensor_sessions(Acc1_preprocessed)
Acc2_matrix = sensor_sessions(Acc2_preprocessed)
Acc3_matrix = sensor_sessions(Acc3_preprocessed)

def label_sessions(logdata,acc_matrix,label):
# labeling for each session   
    ind = np.zeros([len(logdata)/2,2], int)
    j = 0
    for i in range(len(logdata)):
        if np.mod(i,2) == 0:
            ind[j,0] = np.argmin(np.abs(acc_matrix[:,0] - logdata[i,]))
        else:
            ind[j,1] = np.argmin(np.abs(acc_matrix[:,0] - logdata[i,]))
            j = j + 1
    for i in range(ind.shape[0]): 
        label[ind[i,0]:ind[i,1]+1] = 1
    return label

    
label1 = label_sessions(logdata1,Acc1_matrix,label1)
label2= label_sessions(logdata2,Acc2_matrix,label2)
label3 = label_sessions(logdata3,Acc3_matrix,label3) 

# Concatenating all three sessions accelerometer data.
Acc = np.concatenate((Acc1_matrix[:,1:4], Acc2_matrix[:,1:4], Acc3_matrix[:,1:4]), axis=0)  

# concatenating labels of all sessions 
label= np.zeros(Acc.shape[0], int)
label = np.concatenate((label1, label2, label3), axis=0)
  
#---------Save annotations before segmentation-----------------

dataFrame= pd.DataFrame(label, columns=['label'])
dataFrame.to_csv(savePath+'annotation_subject'+str(subject)+'.csv', index= False) 

#---------------------------------------Gyroscope--------------------------
Gyro1_matrix = sensor_sessions(Gyro1_preprocessed)
Gyro2_matrix = sensor_sessions(Gyro2_preprocessed)
Gyro3_matrix = sensor_sessions(Gyro3_preprocessed)
# Concatenating Gyroscope data from 3 sessions
Gyro = np.concatenate((Gyro1_matrix[:,1:4], Gyro2_matrix[:,1:4], Gyro3_matrix[:,1:4]), axis=0)

#---------------------------------------Magnetometer------------------------------
Mag1_matrix = sensor_sessions(Mag1_preprocessed)
Mag2_matrix = sensor_sessions(Mag2_preprocessed)
Mag3_matrix = sensor_sessions(Mag3_preprocessed)

# Concatenating Magnetometer data from 3 sessions
Mag = np.concatenate((Mag1_matrix[:,1:4], Mag2_matrix[:,1:4], Mag3_matrix[:,1:4]), axis=0)

#-------------------------save accelerometer, gyroscope and magnetometer data in a csv file-------------------
Data= np.concatenate((Acc, Gyro, Mag), axis=1)
columns= ['ax','ay','az','gx','gy','gz','mx','my','mz']
dataFrame= pd.DataFrame(Data, columns=columns)
dataFrame.to_csv(savePath+'concatenated_IMU_subject'+str(subject)+'.csv', index= False) 

#-------------Data segmentation----------------------------------
# 
# Loading timestamps*channel data and producing sampleNum*timeInterval*channel*1
# i.e., generating an appropriate format for CNN architecture 

Data= pd.read_csv(savePath+'concatenated_IMU_subject'+str(subject)+'.csv')
label= pd.read_csv(savePath+'annotation_subject'+str(subject)+'.csv')

label=np.squeeze(label.as_matrix())
Data = Data.as_matrix()

sampleNum = Data.shape[0]/samplingFreq*samplingFreq/overlap-10

channelNum = Data.shape[1]
X = np.zeros([sampleNum,samplingFreq,channelNum])

Y = np.zeros([sampleNum,],dtype=int)

for s in range(sampleNum):
    X[s,:,:] = Data[s*overlap:s*overlap+samplingFreq,:]
    if np.mean(label[s*overlap:s*overlap+samplingFreq]) > 0.5:
        Y[s,] = 1

# ----save prepared features for CNN architecture.
sio.savemat(savePath+'subject' + str(subject) + '.mat' ,{'X':X,'Labels':Y})     
#    
