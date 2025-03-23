import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

X=iris.data
y=iris.target

model = RandomForestClassifier(random_state=42)

scores = cross_val_score(model,X,y, cv=5, scoring='accuracy')

print("Her fold için dogruluk skoru:", scores)
print("Ortalama Doğruluk:", scores.mean())

param_grid = {
    'n_estimators': [50,100,150],
    'max_depth': [2,4,6,None],
    'criterion': ['gini','entropy']
}

model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X,y)

print("En İyi dogruluk skoru:", grid_search.best_score_)
print("En İyi paramatreler:",grid_search.best_params_)