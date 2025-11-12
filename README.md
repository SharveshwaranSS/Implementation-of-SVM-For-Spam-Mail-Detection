# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYA B
RegisterNumber: 212224230208
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/content/spam.csv", encoding='ISO-8859-1')


print("🔹 Data Head:")
print(data.head())

print("\n🔹 Data Info:")
print(data.info())

print("\n🔹 Checking for Null Values:")
print(data.isnull().sum())


print("\n🔹 Columns in Dataset:")
print(data.columns)


data = data.iloc[:, :2]
data.columns = ['Category', 'Message']


data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})


X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = SVC(kernel='linear')  
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix for Spam Mail Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


sample_messages = [
    "Congratulations! You have won a free $1000 Walmart gift card. Click here to claim.",
    "Hi John, can we reschedule our meeting for tomorrow morning?",
    "URGENT! Your account has been compromised. Please verify your password immediately!"
]

sample_tfidf = vectorizer.transform(sample_messages)
predictions = model.predict(sample_tfidf)

print("\n🔹 Sample Predictions:")
for msg, label in zip(sample_messages, predictions):
    print(f"Message: {msg}\nPrediction: {'Spam' if label == 1 else 'Ham'}\n")

```

## Output:
### Data head
![alt text](dh.png)

### Data info
![alt text](di.png)

### Checking for Null Values
![alt text](cnv.png)

### Columns in Dataset
![alt text](cd.png)

### Classification Report
![alt text](cr.png)

### Graph
![alt text](gr.png)

### Sample Predictions
![alt text](sr.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
