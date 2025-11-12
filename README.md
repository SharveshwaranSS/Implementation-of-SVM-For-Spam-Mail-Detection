# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: kirthick roshan j
RegisterNumber:  212223040097
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![image](https://github.com/user-attachments/assets/fcb26ff0-4012-4397-b32d-ab4cd516b6c5)

![image](https://github.com/user-attachments/assets/e63a3661-e6b6-4912-90ad-5e0fd686bd4a)

![image](https://github.com/user-attachments/assets/72efd40a-1ebf-4f58-89c9-1dc827f1cf5f)

![image](https://github.com/user-attachments/assets/2cf7e5e9-b263-43a0-a662-a7a8bf543c52)

![image](https://github.com/user-attachments/assets/f7ce46cf-0eea-4c96-a16c-3d561ce721c5)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
