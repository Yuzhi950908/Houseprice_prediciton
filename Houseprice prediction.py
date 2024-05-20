#Digital Sharper Programm：
#Digital Sharper Programm basiert auf zwei Lernunterrichten von Coursera. Diese sind:
#1. Python for Everyone
#2. Machine Learning
#Jeder Teilnehmer muss sich für ein Projekt entscheiden,
#in dem ein ML- Algorithmus verwendet wird. Ich habe mich entschieden,
#den RF-Algorithmus zu benutzen, um Hauspreise vorherzusagen.
#Zusätzlich plane ich, einen Artikel zu schreiben,
#in dem die Details zur Idee und zum Algorithmus beschrieben werden,
#und diesen Artikel auf Medium zu veröffentlichen.
#Leider habe ich das Projekt noch nicht vollständig abgeschlossen.



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

#import train dataset,look at Data and remove the NA vaule
train_dataset = pd.read_csv(r'C:\Users\zheng\Desktop\Kaggle Project\Houseprice Prediction\Raw Data\house-prices-advanced-regression-techniques\train.csv')
train_dataset_new = train_dataset.dropna(axis=1)

# identify which features are text
text_features = train_dataset_new.select_dtypes(include=['object'])

# change them to encode
text_features_encoded = pd.get_dummies(text_features)

# drop the raw text feature
train_dataset_new = train_dataset_new.drop(columns=text_features.columns)

# fill the encode feature into text
train_dataset_new = pd.concat([train_dataset_new, text_features_encoded], axis=1)

#use RFS to get the feature importance
x=train_dataset_new.drop(['Id','SalePrice'],axis=1)
y=train_dataset_new['SalePrice']
rf_model=RandomForestRegressor(n_estimators=100,random_state=46)
fit=rf_model.fit(x,y)

importances = rf_model.feature_importances_
importances = pd.Series(importances)
importances_sorted = importances.sort_values(ascending=False)#from big to small
#choose Top 10
top10_features = importances_sorted.head(10)
top10_features=top10_features[::-1]
top10_feature_labels = x.columns[top10_features.index]
print(top10_feature_labels)
plt.figure(figsize=(10, 8))
plt.barh(top10_feature_labels,top10_features)
plt.xlabel('Importance Score')
plt.ylabel('Features of Houseprice')
plt.title('Feature Importance')
plt.xticks(rotation=90)
plt.tick_params(axis='x', which='both', top=True, bottom=False)
plt.gca().xaxis.set_ticks_position('top')
plt.show()

#I choose only Top 3 Features
#They are 1.OverallQual 2.GrLivArea(Above grade (ground) living area square feet) 3.TotalBsmtSF(Total square feet of basement area)

#Traindataset
train_dataset = train_dataset_new[['OverallQual', 'GrLivArea','TotalBsmtSF']]
print(train_dataset)