from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

timestorun = 10
scores = []
average = 0

def calculate():
  dataset = pd.read_csv("Titanic-Dataset.csv")
  dataset = dataset.drop(columns=["PassengerId", "Cabin", "Name", "Parch", "Ticket", "Embarked"])
  dataset = pd.get_dummies(dataset, columns=["Sex"])

  X = dataset.drop(columns=["Survived"])
  y = dataset["Survived"]

  model = DecisionTreeClassifier(max_depth=5)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  model.fit(X_train, y_train)


  prediction = model.predict(X_test)
  accuracy = accuracy_score(y_test, prediction) * 100

  scores.append(accuracy)

for i in range(timestorun):
 calculate()

for score in scores:
 average += score

average /= timestorun

print("------")
print(average)







