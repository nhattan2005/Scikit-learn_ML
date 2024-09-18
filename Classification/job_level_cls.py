import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.preprocessing import OneHotEncoder

def filter_location(loc):
    result = re.findall("\,\s[A-Z]{2}", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc


data = pd.read_excel("final_project.ods", dtype=str)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
# print(data["career_level"].value_counts())
x = data.drop(target, axis = 1)
y=data[target]

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)



vectorizer = TfidfVectorizer(stop_words="english")
result = vectorizer.fit_transform(x_train["title"])
print(vectorizer.vocabulary_)
print(result.shape)


encoder = OneHotEncoder()
result = encoder.fit_transform(x_train[["location"]])
print(result.shape)