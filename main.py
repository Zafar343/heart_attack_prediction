import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, tree
from sklearn.model_selection import train_test_split
from numpy.random import seed
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras import regularizers
import tensorflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#setting up the seeds for random number generator
seed(1)
tensorflow.random.set_seed(2)

#reading the data in csv file
df = pd.read_csv('Desktop\project\heart.csv')
print(df)

# droping the labels coulmns and storing the features in variable X
X = df.drop(["HeartDisease"], axis = "columns")
#storing the labels in variable y
y = df["HeartDisease"]

#returning dataframe values into nd-array by droping the axes labels
y = y.values

#converting categorial features into dummy features (for converting categorical strings in feature vectors to int/float values)
X = pd.get_dummies(X)
X = X.values
print(y.shape)
print(X.shape)

#To scale the features in a range between 0 and 1
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

#train, val splits
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_scaled, y, test_size=0.3)

#val, test splits
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#to clear any previous keras model in the memory
keras.backend.clear_session()

#-------------NN Model------------------------------------------------------------------------------------
#building sequential nn model
model = Sequential([
    Dense(512, activation='relu',  kernel_regularizer=regularizers.l2(0.01), input_shape=(20,)),
    Dropout(0.3),
    Dense(256, activation='relu',  kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(128, activation='relu',  kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])
model.summary()         #to view the summary of the built model

#Model Training--------------------------------------------------
#training configuration
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#fitting the model on train data
#model train, val accuracy and loss will be stored as history in hist
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=400,
          validation_data=(X_val, Y_val))
##----------------------------------------------------------------

#visualizing the train, val accuracy and loss of the model
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#model evaluation on test data
model_acc = model.evaluate(X_test, Y_test)[1]             #model accuracy is returened for loss use [0]
#print("NN model acc: ",model_acc)

#if you want to predict a single example the use the method below by un-commenting
# #this will return the prediction probability for input example
# x = X_test[0,:]
# #print(x.shape)
# model.predict(x[None,:])

#Confusion matrix is better evaluation for this problem than accuracy
#for confusion matrix we need model predictions
preds = model.predict(X_test)           #predictions on test data (all examples in one go)
#print(preds.shape)
for i in range(len(preds)):
  if preds[i]<0.5:
    preds[i]=0
  else:
    preds[i]=1  
#print("//////NN Model Preds :\n",preds)
nn_conf_matrix = confusion_matrix(Y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=nn_conf_matrix, display_labels=np.array(['No HeartDisease','HeartDisease']))
disp.plot()
plt.title(label="NN Model Confusion Matrix",
          fontsize=16,color="green")
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------

####KNN Classifier--------------------------------------------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors = 9)         #setting knn classifier object (n_neighbors is number of neighbors)
knn.fit(X_train,Y_train)        #fitting Knn classifier on train data
knn_preds = knn.predict(X_test)     #predictions on test data
print("//////knn Preds :\n",knn_preds)
knn_acc =accuracy_score(Y_test,knn_preds)
# print("knn acc: ",knn_acc)

knn_conf_matrix = confusion_matrix(Y_test, knn_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=knn_conf_matrix, display_labels=np.array(['No HeartDisease','HeartDisease']))
disp.plot()
plt.title(label="KNN Classifier Confusion Matrix",
          fontsize=16,color="green")
plt.show()
##-----------------------------------------------------------------------------------------------------------------------------

###SVM Classifier--------------------------------------------------------------------------------------------------------------

svm_classifier = svm.SVC()      #svm classifier object
svm_classifier = svm_classifier.fit(X_train, Y_train)       #fitting on train data
svm_preds = svm_classifier.predict(X_test)              #svm preds
print("//////SVM Preds :\n",svm_preds)
svm_acc = accuracy_score(Y_test,svm_preds)
# print("SVM acc: ",svm_acc)
svm_conf_matrix = confusion_matrix(Y_test, svm_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=svm_conf_matrix, display_labels=np.array(['No HeartDisease','HeartDisease']))
disp.plot()
plt.title(label="SVM Classifier Confusion Matrix",
          fontsize=16, color="green")
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------

##Decision tree classifier
decisionTree = tree.DecisionTreeRegressor()             #decision tree classifier object
decisionTree = decisionTree.fit(X_train, Y_train)       #fitting classifier on training data
decisionTree_preds = decisionTree.predict(X_test)       #preds on test data
print("//////Decision Tree Preds :\n",decisionTree_preds)
decisionTree_acc = accuracy_score(Y_test,decisionTree_preds)
# print("Decision tree acc: ",decisionTree_acc)
decisionTree_conf_matrix = confusion_matrix(Y_test, decisionTree_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=decisionTree_conf_matrix, display_labels=np.array(['No HeartDisease','HeartDisease']))
disp.plot()
plt.title(label="Decision Tree Classifier Confusion Matrix",
          fontsize=16, color="green")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------
##Random forest classifier
RandomForest_clf = RandomForestClassifier(n_estimators = 100)       #classifier object
RandomForest_clf = RandomForest_clf.fit(X_train,Y_train)            #fitting on training data
RandomForest_clf_preds = RandomForest_clf.predict(X_test)           #preds
print("//////RandomForest Preds :\n",RandomForest_clf_preds)
RandomForest_acc = accuracy_score(Y_test,RandomForest_clf_preds)
# print("RandomForest acc: ", RandomForest_acc)
RandomForest_conf_matrix = confusion_matrix(Y_test, RandomForest_clf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=RandomForest_conf_matrix, display_labels=np.array(['No HeartDisease','HeartDisease']))
disp.plot()
plt.title(label="Random Forest Classifier Confusion Matrix",
          fontsize=16,color="green")
plt.show()

print("NN model acc: ",model_acc)
print("knn acc: ",knn_acc)
print("SVM acc: ",svm_acc)
print("Decision tree acc: ",decisionTree_acc)
print("RandomForest acc: ", RandomForest_acc)
#-------------------------------------------------------------------------------------------------------------------------------------------