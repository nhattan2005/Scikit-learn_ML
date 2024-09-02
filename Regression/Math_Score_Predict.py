import pandas as pd
from ydata_profiling import ProfileReport 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title = "StudentScore Report", explorative = True)
# profile.to_file("StudentScore_report.html")


target = "math score"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

nums_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

x_train[["reading score"]] = nums_transformer.fit_transform(x_train[["reading score"]])

education_level = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

genders = x_train["gender"].unique()
lunch = x_train["lunch"].unique()
test_preparation_course = x_train["test preparation course"].unique()

ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_level, genders, lunch, test_preparation_course]))
])

nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers = [
    ("nums_feature", nums_transformer, ["reading score", "writing score"]),
    ("ord_feature", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nominal_transformer, ["race/ethnicity"])
])

result = preprocessor.fit_transform(x_train)

print(result)
