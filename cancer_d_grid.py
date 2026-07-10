from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = load_breast_cancer()

X = df.data
y = df.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(random_state=42)

parameters = {
    'n_estimators' : [50,100,150],
    'max_depth' : [3,5,10]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5
)

grid.fit(X_train, y_train)

print("Best Parameters:\n",grid.best_params_)
print("Best cross validation score:\n",grid.best_score_)