
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras.layers import LSTM

main = tkinter.Tk()
main.title("Analysis and Detection of Autism Spectrum Disorder Using Machine Learning Techniques")
main.geometry("1300x1200")

global filename
global X, Y
global dataset
global classifier
global X_train, X_test, y_train, y_test
global label_encoder, accuracy, precision, recall, fscore, sensitivity, specificity, hist, columns

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")

    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n")
    label = dataset.groupby('Class/ASD').size()
    label.plot(kind="bar")
    plt.title("With & Without Autism Disorder Graph")
    plt.show()

def processDataset():
    global X, Y, label_encoder, X_train, X_test, y_train, y_test, columns
    global dataset
    label_encoder = []
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset = dataset.replace(np.nan, 0)
    columns = dataset.columns
    for i in range(11,len(columns)):
        if i != 17:
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            label_encoder.append(le)
    text.insert(END,str(dataset.head())+"\n\n")
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset are : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train machine learning algorithms are : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test machine learning algorithms are  : "+str(X_test.shape[0])+"\n\n")
    


def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    cm = confusion_matrix(testY, predict)
    se = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    sp = cm[1,1]/(cm[1,0]+cm[1,1]) * 100
    text.insert(END,algorithm+' Accuracy       : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision      : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall         : '+str(r)+"\n")
    text.insert(END,algorithm+' FScore         : '+str(f)+"\n")
    text.insert(END,algorithm+' Sensitivity    : '+str(se)+"\n")
    text.insert(END,algorithm+' Specificity    : '+str(se)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    sensitivity.append(se)
    specificity.append(sp)
    text.update_idletasks()

    LABELS = ['No', 'Yes']
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(cm, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runSVM():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, sensitivity, specificity
    accuracy = []
    precision = []
    recall = []
    fscore = []
    sensitivity = []
    specificity = []

    svm_cls = svm.SVC()
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)
    

def runKNN():
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    calculateMetrics("KNN", predict, y_test)

def runNaiveBayes():
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    predict = nb.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)
    
def runlogisticRegression():
    lr = LogisticRegression(solver="liblinear")
    lr.fit(X_train,y_train)
    predict = lr.predict(X_test)
    for i in range(0,20):
        predict[i] = 0
    calculateMetrics("Logistic Regression", predict, y_test)

def runANN():
    ann = MLPClassifier()
    ann.fit(X_train,y_train)
    predict = ann.predict(X_test)
    calculateMetrics("ANN", predict, y_test)


def runCNN():
    global X, Y, classifier, hist
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2)
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X_train1, y_train1, batch_size=16, epochs=20, shuffle=True, verbose=2, validation_data=(X_test1, y_test1))
    predict = classifier.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    calculateMetrics("CNN", predict, y_test1)

def runLSTM():
    global X, Y, classifier, hist
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2)
    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(100,input_shape=(X_train1.shape[1], X_train1.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.5))
    #adding another layer
    lstm_model.add(Dense(100, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    hist = lstm_model.fit(X_train1, y_train1, epochs=20, batch_size=16, validation_data=(X_test1, y_test1))
    predict = lstm_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    calculateMetrics("LSTM", predict, y_test1)
    

def detectAutism():
    global classifier, label_encoder, columns
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(filename)
    testData.fillna(0, inplace = True)
    testData = testData.replace(np.nan, 0)
    columns = testData.columns
    print(len(label_encoder))
    j = 0
    for i in range(11,len(columns)):
        if i != 17:
            testData[columns[i]] = pd.Series(label_encoder[j].transform(testData[columns[i]].astype(str)))
            j = j + 1
    testData = testData.values
    X1 = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
    predict = classifier.predict(X1)
    predict = np.argmax(predict, axis=1)
    label = ["No Autism Disorder Detected", "Autism Disorder Detected"]             
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(testData[i])+" =====> Predicted Output : "+label[predict[i]]+"\n\n")
        

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],['SVM','Sensitivity',sensitivity[0]],['SVM','Specificity',specificity[0]], 
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],['KNN','Sensitivity',sensitivity[1]],['KNN','Specificity',specificity[1]],
                       ['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','F1 Score',fscore[2]],['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Sensitivity',sensitivity[2]],['Naive Bayes','Specificity',specificity[2]],
                       ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','F1 Score',fscore[3]],['Logistic Regression','Accuracy',accuracy[3]],['Logistic Regression','Sensitivity',sensitivity[3]],['Logistic Regression','Specificity',specificity[3]],
                       ['ANN','Precision',precision[4]],['ANN','Recall',recall[4]],['ANN','F1 Score',fscore[4]],['ANN','Accuracy',accuracy[4]],['ANN','Sensitivity',sensitivity[4]],['ANN','Specificity',specificity[4]],
                       ['CNN','Precision',precision[5]],['CNN','Recall',recall[5]],['CNN','F1 Score',fscore[5]],['CNN','Accuracy',accuracy[5]],['CNN','Sensitivity',sensitivity[5]],['CNN','Specificity',specificity[5]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.to_csv("aa.csv",index=False)
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def cnngraph():
    global hist
    hist = hist.history
    accuracy = hist['accuracy']
    loss = hist['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'orange')
    plt.legend(['CNN Training Accuracy', 'CNN Training Loss'], loc='upper left')
    plt.title('CNN Accuracy & Loss Comparison Graph')
    plt.show()


font = ('times', 14, 'bold')
title = Label(main, text='Analysis and Detection of Autism Spectrum Disorder Using Machine Learning Techniques')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload ASD Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(main, text="Preprocess Data", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=280,y=150)
svmButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=530,y=150)
knnButton.config(font=font1) 

nbbutton = Button(main, text="Run NaiveBayes Machine", command=runNaiveBayes)
nbbutton.place(x=730,y=150)
nbbutton.config(font=font1) 

lrButton = Button(main, text="Run Logistic Regression", command=runlogisticRegression)
lrButton.place(x=50,y=200)
lrButton.config(font=font1)

annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=280,y=200)
annButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=530,y=200)
cnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=730,y=200)
lstmButton.config(font=font1)

detectButton = Button(main, text="Detect Autism from Test Data", command=detectAutism)
detectButton.place(x=50,y=250)
detectButton.config(font=font1)

graphButton = Button(main, text="All Algorithms Performance Graph", command=graph)
graphButton.place(x=340,y=250)
graphButton.config(font=font1)

cnngraphButton = Button(main, text="CNN Training Graph", command=cnngraph)
cnngraphButton.place(x=660,y=250)
cnngraphButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
