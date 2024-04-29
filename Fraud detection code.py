#PREPROCESSING

import pandas as pd
data = pd.read_csv("banktransactions.csv")

data.head()
data.tail()
pd.options.display.max_columns = None
data.shape

(284807, 31)
Number of Rows:     284807
Number of Columns:  31

data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns)
dtypes: float64(30), int64(1)
memory usage: 67.4 MB

data.isnull()

#sum of isnull() true values
data.isnull().sum()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

data = data.drop(['Time'],axis=1)

data.duplicated().any()

True

data = data.drop_duplicates()

data.shape
(275663, 30)

#number of duplicates in the dataset
284807 – 275663
9144

data['Class'].value_counts()

0    275190
1       473
Name: Class, dtype: int64

#-----------------------------------------------------------------------------------------------------------------------
#EVALUATING THE IMBALANCED DATASET

#storing our independent variables in matrix X and target/dependent
#variable in vector y
X = data.drop('Class', axis=1)
y = data['Class']
 
#splitting the dataset into the training set and testing set
#to evaluate the performance of the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#logistic regression algorithm
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
#training the model on the training set
log.fit(X_train,y_train)

#perform prediction
y_pred1 = log.predict(X_test)

#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
0.9992200678359603

#precision score, recall score, f1 score
from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y_test,y_pred1)
0.8870967741935484

recall_score(y_test,y_pred1)
0.6043956043956044

f1_score(y_test,y_pred1)
0.718954248366013

#----------------------------------------------------------------------------------------------------
#BALANCING THE DATASET USING OVERSAMPLING

#Using SMOTE-Synthetic Minority Oversampling Technique;
#randomly increasing minority class values by replicating them 
#synthesizing new minority instances using the existing ones
#(by linear interpolation)

#storing our independent variables in matrix X and target/dependent variable in vector y
X = data.drop('Class', axis=1)
y = data['Class']
X.shape
(275663, 29)

y.shape
(275663,)

#using SMOTE
from imblearn.over_sampling import SMOTE
X_res,y_res = SMOTE().fit_resample(X,y)

#new shape of the target variable
y_res.value_counts()
0    275190
1    275190
Name: Class, dtype: int64
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#DATA SPLITTING AND VALIDATION

#splitting the dataset into the training set and testing set
#to evaluate the performance of the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,random_state=42)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#MODEL SELECTION

#LogisticRegression after oversampling
log = LogisticRegression()
log.fit(X_train,y_train)
#prediction
y_pred4 = log.predict(X_test) 

#accuracy score, precision score, recall score, f1 score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#(actual value, predicted value)
accuracy_score(y_test,y_pred4)
0.944583742141793

precision_score(y_test,y_pred4)
0.9734167166837693

recall_score(y_test,y_pred4)
0.9140592331327382

f1_score(y_test,y_pred4)
0.9428046356374001

#DecisionTreeClassifier after oversampling
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

#prediction
y_pred5 = dt.predict(X_test)

accuracy_score(y_test,y_pred5)
0.9980740579236164

precision_score(y_test,y_pred5)
0.9973133407155953

recall_score(y_test,y_pred5)
0.9988364271039761

f1_score(y_test,y_pred5)
0.9980743028431283

#RandomForestClassifier after oversampling
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

#prediction
y_pred6 = rf.predict(X_test)

accuracy_score(y_test,y_pred6)
0.9999091536756423

precision_score(y_test,y_pred6)
0.999818224783233

recall_score(y_test,y_pred6)
1.0

f1_score(y_test,y_pred6)
0.9999091041303083

#--------------------------------------------------------------------------------------------------------
#VISUALIZE THE RESULTS

accuracy_data = pd.DataFrame({'Models':['LR', 'DT', 'RF'], "ACC":[accuracy_score(y_test,y_pred4)*100, accuracy_score(y_test,y_pred5)*100, accuracy_score(y_test,y_pred6)*100]})

accuracy_data


Models	ACC
0	LR	94.458374
1	DT	99.807406
2	RF	99.990915

 
precision_data = pd.DataFrame({'Models':['LR', 'DT', 'RF'], "PRE":[precision_score(y_test,y_pred4)*100, precision_score(y_test,y_pred5)*100, precision_score(y_test,y_pred6)*100]})

precision_data


Models	  PRE
0	LR	97.341672
1	DT	99.731334
2	RF	99.981822



recall_data = pd.DataFrame({'Models':['LR', 'DT', 'RF'], "REC":[recall_score(y_test,y_pred4)*100, recall_score(y_test,y_pred5)*100, recall_score(y_test,y_pred6)*100]})

recall_data

Models	REC
0	LR	91.405923
1	DT	99.883643
2	RF	100.000000


f1_data = pd.DataFrame({'Models':['LR', 'DT', 'RF'], "F1":[f1_score(y_test,y_pred4)*100, f1_score(y_test,y_pred5)*100, f1_score(y_test,y_pred6)*100]})

f1_data

	Models	F1
0	LR	94.280464
1	DT	99.807430
2	RF	99.990910

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#MODEL TRAINING AND METRICS EVALUATION

#Training RandomForestClassifier on entire dataset after oversampling
rf = RandomForestClassifier()
rf.fit(X_res,y_res)

#prediction
y_pred6 = rf.predict(X_test)

accuracy_score(y_test,y_pred6)
0.9999091536756423

precision_score(y_test,y_pred6)
0.999818224783233

recall_score(y_test,y_pred6)
1.0

f1_score(y_test,y_pred6)
0.9999091041303083

confusion_matrix(y_test,y_pred6)
array([[55065,     8],
       [    0, 55003]], dtype=int64)

roc_auc_score(y_test, probabilities)
0.999999727153903

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#SIMULATION AND MODEL IMPLEMENTATION

#saving the model
import joblib
joblib.dump(rf,"fraud_detection_model")
['fraud_detection_model']

#Using tkinter to implement the GUI.
from tkinter import *
import joblib

def show_entry_fields():
    v1=float(e1.get())
    v2=float(e2.get())
    v3=float(e3.get())
    v4=float(e4.get())
    v5=float(e5.get())
    v6=float(e6.get()) 
    v7=float(e7.get())
    v8=float(e8.get())
    v9=float(e9.get())
    v10=float(e10.get())
    v11=float(e11.get())
    v12=float(e12.get())
    v13=float(e13.get())
    v14=float(e14.get())
    v15=float(e15.get())
    v16=float(e16.get())
    v17=float(e17.get())
    v18=float(e18.get())
    v19=float(e19.get())
    v20=float(e20.get())
    v21=float(e21.get())
    v22=float(e22.get())
    v23=float(e23.get())
    v24=float(e24.get())
    v25=float(e25.get())
    v26=float(e26.get())
    v27=float(e27.get())
    v28=float(e28.get())
    v29=float(e29.get())
    
    model = joblib.load("fraud_detection_model")
    y_pred = model.predict([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]])
    list1 = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]
    
       
    result = []
    if y_pred == 0:
        result.append("NORMAL TRANSACTION")
    else:
        result.append("FRAUD TRANSACTION")
    print("######################################") 
    print("Fraud Detection System", result)
    print("######################################") 
    
    
    Label(root, text = "Final Prediction:",bg="lightblue",font=("Segoe Script",10,"underline","bold")).grid(row=31,column=1)
    Label(root,text = result,bg="lightblue",font=("Segoe Script",10,"bold")).grid(row=32,column=1)
    
root = Tk()
root.title("Fraud Detection System")
root.configure(bg="lightblue")
root.minsize(300,700)
root.maxsize(300,700)
 
Label(root, text="Enter value of V1",bg="lightblue").grid(row=1)
Label(root, text="Enter value of V2",bg="lightblue").grid(row=2)
Label(root, text="Enter value of V3",bg="lightblue").grid(row=3)
Label(root, text="Enter value of V4",bg="lightblue").grid(row=4)
Label(root, text="Enter value of V5",bg="lightblue").grid(row=5)
Label(root, text="Enter value of V6",bg="lightblue").grid(row=6)
Label(root, text="Enter value of V7",bg="lightblue").grid(row=7)
Label(root, text="Enter value of V8",bg="lightblue").grid(row=8)
Label(root, text="Enter value of V9",bg="lightblue").grid(row=9)
Label(root, text="Enter value of V10",bg="lightblue").grid(row=10)
Label(root, text="Enter value of V11",bg="lightblue").grid(row=11)
Label(root, text="Enter value of V12",bg="lightblue").grid(row=12)
Label(root, text="Enter value of V13",bg="lightblue").grid(row=13)
Label(root, text="Enter value of V14",bg="lightblue").grid(row=14)
Label(root, text="Enter value of V15",bg="lightblue").grid(row=15)
Label(root, text="Enter value of V16",bg="lightblue").grid(row=16)
Label(root, text="Enter value of V17",bg="lightblue").grid(row=17)
Label(root, text="Enter value of V18",bg="lightblue").grid(row=18)
Label(root, text="Enter value of V19",bg="lightblue").grid(row=19)
Label(root, text="Enter value of V20",bg="lightblue").grid(row=20)
Label(root, text="Enter value of V21",bg="lightblue").grid(row=21)
Label(root, text="Enter value of V22",bg="lightblue").grid(row=22)
Label(root, text="Enter value of V23",bg="lightblue").grid(row=23)
Label(root, text="Enter value of V24",bg="lightblue").grid(row=24)
Label(root, text="Enter value of V25",bg="lightblue").grid(row=25)
Label(root, text="Enter value of V26",bg="lightblue").grid(row=26)
Label(root, text="Enter value of V27",bg="lightblue").grid(row=27)
Label(root, text="Enter value of V28",bg="lightblue").grid(row=28)
Label(root, text="Enter value of V29",bg="lightblue").grid(row=29)


e1 = Entry(root, width=20)
e2 = Entry(root, width=20)
e3 = Entry(root, width=20)
e4 = Entry(root, width=20)
e5 = Entry(root, width=20)
e6 = Entry(root, width=20)
e7 = Entry(root, width=20)
e8 = Entry(root, width=20)
e9 = Entry(root, width=20)
e10 = Entry(root, width=20)
e11 = Entry(root, width=20)
e12 = Entry(root, width=20)
e13 = Entry(root, width=20)
e14 = Entry(root, width=20)
 
e15 = Entry(root, width=20)
e16 = Entry(root, width=20)
e17 = Entry(root, width=20)
e18 = Entry(root, width=20)
e19 = Entry(root, width=20)
e20 = Entry(root, width=20)
e21 = Entry(root, width=20)
e22 = Entry(root, width=20)
e23 = Entry(root, width=20)
e24 = Entry(root, width=20)
e25 = Entry(root, width=20)
e26 = Entry(root, width=20)
e27 = Entry(root, width=20)
e28 = Entry(root, width=20)
e29 = Entry(root, width=20)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)
e14.grid(row=14, column=1)
e15.grid(row=15, column=1)
e16.grid(row=16, column=1)
e17.grid(row=17, column=1)
e18.grid(row=18, column=1)
e19.grid(row=19, column=1)
e20.grid(row=20, column=1)
e21.grid(row=21, column=1)
e22.grid(row=22, column=1)
e23.grid(row=23, column=1)
e24.grid(row=24, column=1)
e25.grid(row=25, column=1)
e26.grid(row=26, column=1)
e27.grid(row=27, column=1)
e28.grid(row=28, column=1)
e29.grid(row=29, column=1)
 
Button(root, text="Predict", command=show_entry_fields, bg="lightgreen",
      activebackground="lightblue", fg="black").grid(row=30, column=1) 
                
root.mainloop()
