
"""
Nastaran Mohammadian Rad <nastaran@fbk.eu>
Paper: "Stereotypical Motor Movement Detection in Dynamic Feature Space"
LSTM on learned features by CNN.
Experiment: Dynamic-Features-Unbalanced on Simulated Dataset.
"""
import numpy as np
import scipy.io as sio
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
import keras
import pickle
# set input and output paths
path = '~/smm_detection/Results/CNN_Unbalanced/'
savePath = '~/smm_detection/Results/CNN_LSTM/'

# variables definition
study = 'Simulated_'
subNum = [0,1,2,3,4,5]
runNum = 10
batch_size = 100
nb_classes = 2 #binary classification
s = 0
ts = [1,3,5,10,15,25,50] # time steps
neuronNum = [5,10,20,40,50,100] 

AUCLSTM = np.zeros([len(subNum),runNum,len(ts),len(neuronNum)])
f1LSTM = np.zeros([len(subNum),runNum,len(ts),len(neuronNum)])
precisionLSTM = np.zeros([len(subNum),runNum,len(ts),len(neuronNum)])
recallLSTM = np.zeros([len(subNum),runNum,len(ts),len(neuronNum)])
MCCLSTM = np.zeros([len(subNum),runNum,len(ts),len(neuronNum)])
fpr = dict()
tpr = dict()
roc_auc = dict()


for sub in subNum:
    for run in range(runNum):
        #reading learned features by CNN
        matContent = sio.loadmat(path + study +'CNN_learned_features'+ '_sub_' + str(sub+1) + '_run_' + str(run+1) + '.mat') 
        trainingFeatures = matContent['trainingFeatures']
        trainingLabels = matContent['trainingLabels']
        testFeatures = np.transpose(matContent['testFeatures'])
        testLabels = matContent['testLabels']
        del matContent
        
        testLabels = np.transpose(testLabels)        
        testFeatures = np.transpose(testFeatures)        
        #Normalization
        scaler = StandardScaler()
        scaler.fit(trainingFeatures)
        trainingFeatures = scaler.transform(trainingFeatures)
        testFeatures = scaler.transform(testFeatures)
        t_idx = 0
        for timesteps in ts:
            lstm_training_features = np.zeros([trainingFeatures.shape[0]-timesteps+1,timesteps,trainingFeatures.shape[1]]) 
            lstm_test_features = np.zeros([testFeatures.shape[0]-timesteps+1,timesteps,testFeatures.shape[1]]) 
            for i in range(trainingFeatures.shape[0]-timesteps+1):
                lstm_training_features[i,:,:] = trainingFeatures[i:i+timesteps,:]
            for i in range(testFeatures.shape[0]-timesteps+1):
                lstm_test_features[i,:,:] = testFeatures[i:i+timesteps,:]
            n_idx = 0    
            for n in neuronNum:
                # LSTM architecture
                
                lstm_model= Sequential()
                lstm_model.add(LSTM(n, init='glorot_normal', consume_less = 'gpu', return_sequences=False,input_shape=(timesteps, trainingFeatures.shape[1])))  # returns a sequence of vectors of dimension 32
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(2, activation='softmax'))
                lstm_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
                lstm_model.fit(lstm_training_features, trainingLabels[timesteps-1:,:], 
                               batch_size=batch_size, nb_epoch=15, verbose=2,shuffle=False,callbacks=[earlyStopping],validation_split=0.1)
                               
                predicted_labels_LSTM=np.float64(lstm_model.predict_classes(lstm_test_features,verbose=0))                
                precisionLSTM[s,run,t_idx,n_idx] = precision_score(testLabels[timesteps-1:],predicted_labels_LSTM)
                # evaluation 
                recallLSTM[s,run,t_idx,n_idx] = recall_score(testLabels[timesteps-1:],predicted_labels_LSTM)    
                f1LSTM[s,run,t_idx,n_idx] = f1_score(testLabels[timesteps-1:],predicted_labels_LSTM) 
                MCCLSTM[s,run,t_idx,n_idx] = matthews_corrcoef(testLabels[timesteps-1:],predicted_labels_LSTM)  
                print('Subject %d : Run %d : Timesteps %d : NeuronNum %d :F1_Score_LSTM: %.4f' % (sub+1, run+1, timesteps, n, f1LSTM[s,run,t_idx,n_idx]))   
                soft_targets_test_LSTM=lstm_model.predict(lstm_test_features)
                AUCLSTM[s,run,t_idx,n_idx] = roc_auc_score(testLabels[timesteps-1:], soft_targets_test_LSTM[:,1])
                print('Subject %d : Run %d : Timesteps %d : NeuronNum %d : AUC_LSTM: %.4f' % (sub+1, run+1, timesteps, n, AUCLSTM[s,run,t_idx,n_idx]))        
                fpr[s,run,t_idx,n_idx], tpr[s,run,t_idx,n_idx], _ = roc_curve(testLabels[timesteps-1:],soft_targets_test_LSTM[:,1])
                roc_auc[s,run,t_idx,n_idx] = auc(fpr[s,run,t_idx,n_idx], tpr[s,run,t_idx,n_idx])
                
                #save the model and weights
                json_string = lstm_model.to_json()
                open(savePath + study + 'LSTM_prediction' + str(sub+1) + '_Run_' + str(run+1) + '_TS_' + str(timesteps) + '_NN_' + str(n) +'.json', 'w').write(json_string)       
                lstm_model.save_weights(savePath + study + 'LSTM_prediction' + str(sub+1) + '_Run_' + str(run+1) + '_TS_' + str(timesteps) + '_NN_' + str(n) + '.h5', overwrite=True)
                #save results
                
                sio.savemat(savePath + study + 'CNN_LSTM_Prediction_Sub_' + str(sub+1) + '.mat' ,{'f1LSTM':f1LSTM,'AUCLSTM':AUCLSTM,'precisionLSTM':precisionLSTM,'recallLSTM':recallLSTM,'MCCLSTM':MCCLSTM})
                pickle.dump( fpr, open(savePath + 'CNN_LSTM_Prediction_pickle_fpr_Sub_' + str(sub+1) + '.p' , "wb" ) )
                pickle.dump( tpr, open(savePath + 'CNN_LSTM_Prediction_pickle_tpr_Sub_' + str(sub+1) + '.p', "wb" ) )
                pickle.dump( roc_auc, open(savePath + 'CNN_LSTM_Prediction_pickle_roc_Sub_' + str(sub+1) + '.p', "wb" ) )                 
                
                n_idx += 1
            t_idx += 1
    s += 1
  





