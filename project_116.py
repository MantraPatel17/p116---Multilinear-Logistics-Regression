
import pandas as pd
import plotly.express as px

df = pd.read_csv('data.csv')
toefl_list = df["TOEFL Score"].tolist()
ca_list = df["Chance of admit"].tolist()

fig = px.scatter(x=toefl_list,y=ca_list)
fig.show()

import plotly.graph_objects as go

age_list = df["GRE Score"].tolist()
colors = []
for i in ca_list :
  if i == 1 :
    colors.append("green")
  else :
    colors.append("red")

fig = go.Figure(data = go.Scatter(
    x = toefl_list,
    y = age_list,
    mode = 'markers',
    marker = dict(color = colors)
))

fig.show()

factors = df[["TOEFL Score","GRE Score"]]
purchases = df["Chance of admit"]

from sklearn.model_selection import train_test_split

salary_train,salary_test,purchase_train,purchase_test = train_test_split(factors,purchases,test_size = 0.25,random_state = 0)

print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
salary_train = sc.fit_transform(salary_train)
salary_test = sc.fit_transform(salary_test)

print(salary_train[0:10])

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(salary_train,purchase_train)

purchase_pred = classifier.predict(salary_test)
from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(purchase_test,purchase_pred))

age = int(input("enter gre score = "))
salary = int(input("enter toefl = "))
user_test = sc.transform([[salary,age]])
user_pred = classifier.predict(user_test)
if user_pred[0]==1 : 
  print("You will be fine😁😁😁😁")
else: 
  print("Sorry to say but you fail")