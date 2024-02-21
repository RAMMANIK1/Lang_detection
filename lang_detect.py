import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Preprocessing
x = np.array(data["Text"])
y = np.array(data["language"])

# CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit app
st.title('Language Prediction')

text_input = st.text_area("Enter text:", "")
if st.button("Predict"):
    text_transformed = cv.transform([text_input])
    prediction = model.predict(text_transformed)
    st.write('Predicted language:', prediction[0])
