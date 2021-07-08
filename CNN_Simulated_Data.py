
"""
@author: Nastaran Mohammadian Rad <Email: nastaran@fbk.eu>
Paper: "Stereotypical Motor Movement Detection in Dynamic Feature Space"
To reproduce the results of the Static-Features-Unbalanced and Static-Features-Balanced experiments on the simulated data. 
"""

import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import f1_score,  auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.utils import np_utils 
from deeppy import StandardScaler
import keras
from keras import backend as K

# Setting Network Parameters
nb_filters=[4, 4, 8]
hidden_neuron_num = 8
filter_size = [10, 1]
channels = 9
pool_size = (3,1)
strides=(2,1)

# Setting Learning Parameters
subNum = 5 
runNum = 10
nb_epoch = [10,5] 
learn_rate = 0.05
batch_size = 100
mtm=0.9
nb_classes=2
backend = 'tf'
borderMode = 'same'
balanced = 0 # unbalanced-CNN experiment.

f1Net = np.zeros([runNum,subNum])
accNet = np.zeros([runNum,subNum])
AUCNet = np.zeros([runNum,subNum])
precisionNet = np.zeros([runNum,subNum])
recallNet = np.zeros([runNum,subNum])
MCCNet = np.zeros([runNum,subNum])
fpr = dict()
tpr = dict()
roc_auc = dict()
study = 'Simulated_'
temp_X = []
temp_Y = []  
# set the output and input paths
path = '~/smm-detection/Simulated_Data/Results/Intermediate_files/'  
savePath = '~/smm-detection/Simulated_Data/Results/CNN_Unbalanced/' # save the results of unbalanced-CNN
#savePath = '~/smm_detection/Simulated_Data/Results/CNN_balanced/' #save the results of balanced-CNN

subjects= ["subject1","subject2","subject3","subject4","subject5"]
# reading data
for i in range (np.shape(subjects)[0]):
    matContent = sio.loadmat(path + subjects[i] + '.mat')
    temp = matContent['X']
    temp_X.append(temp)
    temp = matContent['Labels']
    temp_Y.append(temp)

# creating training and test dataset based on one-subject-leave-out scenario.
for sub in range(subNum): 
    testFeatures = temp_X[sub]
    testLabels = temp_Y[sub]
    testLabels = testLabels.astype(int)
    testLabels =  np.squeeze(testLabels)
    
    train_index = np.setdiff1d(range(subNum),sub)
    trainingFeatures = temp_X[train_index[0]]
    trainingLabels = temp_Y[train_index[0]]
    train_index = np.setdiff1d(train_index,train_index[0])
    for j in range(len(train_index)):
        trainingFeatures = np.concatenate((trainingFeatures,temp_X[train_index[j]]),axis = 0)
        trainingLabels = np.concatenate((trainingLabels,temp_Y[train_index[j]]),axis = 1)
    
    if balanced: # To replicate the results of CNN on balanced data or Static-Features-Balanced experiment
        t = np.sum(trainingLabels,axis = 1)
        temp_smm = trainingFeatures[trainingLabels[0,:]==1,:,:]
        temp_nosmm = trainingFeatures[trainingLabels[0,:]==0,:,:]
        idx = np.random.permutation(temp_nosmm.shape[0])
        idx = idx[:t]
        temp_nosmm = temp_nosmm[idx,:,:]
        trainingFeatures = np.concatenate((temp_nosmm,temp_smm),axis = 0)
        trainingLabels = np.zeros([1,2*t])
        trainingLabels[0,t:]=1
        idx = np.random.permutation(trainingFeatures.shape[0])
        trainingFeatures = trainingFeatures[idx,:,:]
        trainingLabels = trainingLabels[0,idx]    
    
    # constructing data compatible with tensorflow backend
    trainingFeatures = np.transpose(trainingFeatures, (0, 2, 1)) # Chanel should be in the second dimension
    trainingFeatures = np.float64(trainingFeatures[..., np.newaxis]) #
    trainingFeatures = np.transpose(trainingFeatures, (0, 2, 3,1))
    testFeatures = np.transpose(testFeatures, (0, 2, 1)) 
    testFeatures = np.float64(testFeatures[..., np.newaxis])
    testFeatures = np.transpose(testFeatures, (0, 2, 3,1))
        
    trainingLabels = trainingLabels.astype(int)
    trainingLabels =  np.squeeze(trainingLabels)         

    trainingLabels = np_utils.to_categorical(trainingLabels, nb_classes)
    testLabels1 = np_utils.to_categorical(testLabels, nb_classes)

    # Normalization
    scaler = StandardScaler() 
    scaler.fit(trainingFeatures)
    trainingFeatures = scaler.transform(trainingFeatures)
    testFeatures = scaler.transform(testFeatures)
    
    for run in range(runNum):
        # Prepare network inputs
        model= Sequential()

        model.add(Convolution2D(nb_filters[0],filter_size[0], filter_size[1], border_mode=borderMode, dim_ordering=backend
                                ,input_shape=(trainingFeatures.shape[1], trainingFeatures.shape[2],trainingFeatures.shape[3]), init='he_normal'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size, strides, border_mode=borderMode,dim_ordering=backend ))

        model.add(Convolution2D(nb_filters[1], filter_size[0], filter_size[1], border_mode=borderMode, dim_ordering=backend, init='he_normal'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend))

        model.add(Convolution2D(nb_filters[2], filter_size[0], filter_size[1], border_mode=borderMode, dim_ordering=backend, init='he_normal'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size, strides, border_mode=borderMode, dim_ordering=backend))

        model.add(Flatten())
        
        model.add(keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
        
        model.add(Dense(8, init='he_normal'))
        model.add(Dropout(0.2))

        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('softmax'))
        earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

        for ep in range(len(nb_epoch)):
            sgd=SGD(lr=learn_rate/10**ep, momentum=0.9, decay=0.03, nesterov=True)
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            model.fit(trainingFeatures, trainingLabels, batch_size=batch_size, nb_epoch=nb_epoch[ep],
             verbose=2, callbacks=[earlyStopping], validation_split=0.1)
                        
       # Saving the learned features from the flattening layer
       
        get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[9].output,])
        layer_output_training = get_3rd_layer_output([trainingFeatures])[0] 
        layer_output_test = get_3rd_layer_output([testFeatures])[0]
        sio.savemat(savePath + study + 'CNN_learned_features' + '_sub_'+str(sub+1) +'_run_'+ str(run+1)+ '.mat', {'trainingFeatures':layer_output_training, 'trainingLabels':trainingLabels,
            'testFeatures':layer_output_test, 'testLabels':testLabels})       
        # Prediction   
        predicted_labelsNet=np.float64(model.predict_classes(testFeatures),verbose = 0)
        soft_targets_train = model.predict(trainingFeatures)
        soft_targets_test = model.predict(testFeatures)
        
        # evaluation 
        precisionNet[run,sub] = precision_score(testLabels,predicted_labelsNet)
        recallNet[run,sub] = recall_score(testLabels,predicted_labelsNet)    
        f1Net[run,sub] = f1_score(testLabels,predicted_labelsNet) 
        MCCNet[run,sub] = matthews_corrcoef(testLabels,predicted_labelsNet)        
        print('Subject %d : Run %d :F1_Score_Net: %.4f' % (sub+1, run+1, f1Net[run,sub]))   
        AUCNet[run,sub] = roc_auc_score(testLabels, soft_targets_test[:,1])
        print('Subject %d : Run %d :AUC_Net: %.4f' % (sub+1, run+1, AUCNet[run,sub]))        
        fpr[run,sub], tpr[run,sub], _ = roc_curve(testLabels, soft_targets_test[:,1])
        roc_auc[run,sub] = auc(fpr[run,sub], tpr[run,sub])        
        
        # save the model and weights
        json_string = model.to_json()
        open(savePath + study + 'CNN' + str(sub+1) + '_Run_' + str(run+1) + '.json', 'w').write(json_string)       
        model.save_weights(savePath + study + 'CNN' + str(sub+1) + '_Run_' + str(run+1) + '.h5', overwrite=True)  
        # save results
        sio.savemat(savePath + study + 'CNN_Results' + '.mat' ,{'precisionNet':precisionNet,
        'recallNet':recallNet,'f1Net':f1Net,'AUCNet':AUCNet, 'MCCNet':MCCNet})       


pickle.dump( fpr, open(savePath + 'CNN_pickle_fpr.p' , "wb" ) )
pickle.dump( tpr, open(savePath + 'CNN_pickle_tpr.p', "wb" ) )
pickle.dump( roc_auc, open(savePath + 'CNN_pickle_roc.p', "wb" ) ) 

