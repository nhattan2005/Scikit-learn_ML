import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title="Student score Report", explorative=True)
# profile.to_file("student_report.html")
target = "math score"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_levels = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

genders = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()
prep_courses = x_train["test preparation course"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, prep_courses])),
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder()),
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "writing score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])
# x_train = reg.fit_transform(x_train)
# x_test = reg.transform(x_test)

params = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
    "regressor__max_depth": [None, 2, 5],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 5],
    "preprocessor__num_feature__imputer__strategy": ["mean", "median"]
}

model = RandomizedSearchCV(
    estimator=reg,
    param_distributions=params,
    n_iter=30,
    scoring="r2",
    cv=6,
    verbose=2,
    n_jobs=6
)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)
# y_predict = model.predict(x_test)
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))
reg = LazyRegressor(verbose=2, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)
