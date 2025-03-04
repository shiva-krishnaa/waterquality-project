from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout

import keras.layers
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


global X, Y, dataset, X_train, X_test, y_train, y_test
global algorithms, accuracy, f1, precision, recall, classifier

np.set_printoptions(suppress=True)

def ProcessData(request):
    if request.method == 'GET':
        global X, Y, dataset, X_train, X_test, y_train, y_test
        dataset = pd.read_csv("Dataset/ml.csv")
        dataset.fillna(0, inplace = True)
        label = dataset.groupby('labels').size()
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:,2:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th>"+font+columns[i]+"</th>"            
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr>"
            for j in range(0,temp.shape[1]):
                output += '<td><font size="" color="black">'+str(temp[i,j])+'</td>'
            output += "</tr>"    
        context= {'data': output}
        label.plot(kind="bar")
        plt.title("Water Quality Graph, 0 (Good quality) & 1 (Poor Quality)")
        plt.show()
        return render(request, 'UserScreen.html', context)

def Forecast(request):
    if request.method == 'GET':
        dataset = pd.read_csv("Dataset/ml.csv",usecols=['tds','turbidty','ph','conductivity','temperature'])
        dataset.fillna(0, inplace = True)
        Y = dataset.values[:,1:2]
        Y = Y.reshape(Y.shape[0],1)
        dataset.drop(['turbidty'], axis = 1,inplace=True)
        X = dataset.values
        sc = MinMaxScaler(feature_range = (0, 1))
        X = sc.fit_transform(X)
        Y = sc.fit_transform(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
        #training with rabdom forest
        rf_regression = RandomForestRegressor()
        rf_regression.fit(X_train, y_train.ravel())
        predict = rf_regression.predict(X_test)
        predict = predict.reshape(predict.shape[0],1)
        predict = sc.inverse_transform(predict)
        predict = predict.ravel()
        labels = sc.inverse_transform(y_test)
        labels = labels.ravel()
        print("Predicted Growth: "+str(predict))
        print("\nOriginal Growth: "+str(labels))

        arr = ['Test Water Turbidty', 'Forecast Water Turbidity', 'Forecast Water Quality']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(predict)):
            if predict[i] < 1:
                output +="<tr><td>"+font+str(labels[i])+"</td><td>"+font+str(predict[i])+"</td><td>"+font+"Clean Water"+"</td></tr>"
            else:
                output +="<tr><td>"+font+str(labels[i])+"</td><td>"+font+str(predict[i])+"</td><td>"+font+"Dirty Water"+"</td></tr>"
            
        plt.plot(labels, color = 'red', label = 'Current Water Turbidty')
        plt.plot(predict, color = 'green', label = 'Forecast Water Turbidty')
        plt.title('Water Quality Forecasting')
        plt.xlabel('Test Data Quality')
        plt.ylabel('Forecasting Quality')
        plt.legend()
        plt.show()
        context= {'data': output}
        return render(request, 'UserScreen.html', context)


def TrainRF(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall, classifier
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        cls = RandomForestClassifier()
        cls.fit(X, Y)
        classifier = cls
        predict = cls.predict(X_test)
        p = round(precision_score(y_test, predict,average='macro') * 100,2)
        r = round(recall_score(y_test, predict,average='macro') * 100,2)
        f = round(f1_score(y_test, predict,average='macro') * 100,2)
        a = round(accuracy_score(y_test,predict)*100,2)
        algorithms.append("Random Forest (Proposed System)")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainLSTM(request):
    if request.method == 'GET':
        global X, Y
        global algorithms, accuracy, fscore, precision, recall
        algorithms = []
        accuracy = []
        fscore = []
        precision = []
        recall = []
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y1 = to_categorical(Y)
        print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
        if request.method == 'GET':
            lstm_model = Sequential()
            lstm_model.add(keras.layers.LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
            lstm_model.add(Dropout(0.5))
            lstm_model.add(Dense(100, activation='relu'))
            lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
            lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            lstm_model.fit(X1, Y1, epochs=40, batch_size=32, validation_data=(X_test, y_test))             
            print(lstm_model.summary())#printing model summary
            predict = lstm_model.predict(X_test)
            predict = np.argmax(predict, axis=1)
            testY = np.argmax(y_test, axis=1)
            p = round(precision_score(testY, predict,average='macro') * 100,2)
            r = round(recall_score(testY, predict,average='macro') * 100,2)
            f = round(f1_score(testY, predict,average='macro') * 100,2)
            a = round(accuracy_score(testY,predict)*100,2)
            algorithms.append("LSTM")
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            output = '<table border=1 align=center width=100%>'
            font = '<font size="" color="black">'
            output += "<tr>"
            for i in range(len(arr)):
                output += "<th>"+font+arr[i]+"</th>"
            output += "</tr>"
            for i in range(len(algorithms)):
                output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
            context= {'data': output}
            return render(request, 'UserScreen.html', context)
        
def TrainSVM(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        svm = SVC(kernel='rbf', C=1)
        svm.fit(X_train, y_train)
        predict = svm.predict(X_test)
        p = round(precision_score(y_test, predict, average='macro') * 100,2)
        r = round(recall_score(y_test, predict, average='macro') * 100,2)
        f = round(f1_score(y_test, predict, average='macro') * 100,2)
        a = round(accuracy_score(y_test, predict) * 100,2)
        algorithms.append("Support Vector Machine")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)
    
def TrainGBC(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        gbc = GradientBoostingClassifier(n_estimators=100)
        gbc.fit(X_train, y_train)
        predict = gbc.predict(X_test)
        p = round(precision_score(y_test, predict, average='macro') * 100,2)
        r = round(recall_score(y_test, predict, average='macro') * 100,2)
        f = round(f1_score(y_test, predict, average='macro') * 100,2)
        a = round(accuracy_score(y_test, predict) * 100,2)
        algorithms.append("Gradient Boosting")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainKNN(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        p = round(precision_score(y_test, predict, average='macro') * 100,2)
        r = round(recall_score(y_test, predict, average='macro') * 100,2)
        f = round(f1_score(y_test, predict, average='macro') * 100,2)
        a = round(accuracy_score(y_test, predict) * 100,2)
        algorithms.append("K-Nearest Neighbors")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)


def TrainDTC(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        predict = dt.predict(X_test)
        p = round(precision_score(y_test, predict, average='macro') * 100,2)
        r = round(recall_score(y_test, predict, average='macro') * 100,2)
        f = round(f1_score(y_test, predict, average='macro') * 100,2)
        a = round(accuracy_score(y_test, predict) * 100,2)
        algorithms.append("Decision Tree")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)


def TrainLR(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        predict = lr.predict(X_test)
        p = round(precision_score(y_test, predict, average='macro') * 100,2)
        r = round(recall_score(y_test, predict, average='macro') * 100,2)
        f = round(f1_score(y_test, predict, average='macro') * 100,2)
        a = round(accuracy_score(y_test, predict) * 100,2)
        algorithms.append("Logistic Regression")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        testFile = request.POST.get('t1', False)
        test_df = pd.read_csv("Dataset/testData.csv")
        test_df.fillna(0, inplace=True)
        X = test_df.iloc[:, 2:7].values 
        predict = classifier.predict(X)
        print(predict)
        arr = ['Test Data', 'Water Quality Forecasting Result']
        output = '<table border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th>"+font+arr[i]+"</th>"
        output += "</tr>"
        labels = ['Good Quality', 'Poor Quality']
        for i in range(len(predict)):
            output +="<tr><td>"+font+str(test_df.values[i])+"</td><td>"+font+str(labels[predict[i]])+"</td></tr>"
        context = {'data': output}
        return render(request, 'UserScreen.html', context)


def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break
        if index == 1:
            context= {'data':'Welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Login failed. Please retry'}
            return render(request, 'UserLogin.html', context)

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'Waterquality',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      


