import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier
from ydata_profiling import ProfileReport


data = pd.read_csv("diabetes.csv")
profile = ProfileReport(data, title = "Diabetes Report")
profile.to_file("Diabetes_report.html")

target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)

GridSearch: 
params = {
    "n_estimators" : [50, 100, 200],
    "criterion" : ["gini", "entropy", "log_loss"]
}
clf = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = params,
    scoring = "recall",
    cv = 6,
    verbose = 1
    )

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)
print(classification_report(y_test, y_predict))
