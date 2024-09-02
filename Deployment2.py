import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load the data
df = pd.read_csv("E:\\download folder\\Hotel_Reviews.csv", encoding='latin1')

# Preprocessing
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(df["Review"]).toarray()

LE = LabelEncoder()
y = LE.fit_transform(df["Feedback"])

# Initialize and train the Logistic Regression model on the entire dataset
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(x, y)

# Streamlit App
st.title("Hotel Reviews Sentiment Analysis")

st.write("""
### Enter a review to predict whether it is Positive or Negative:
""")

# Input text from user
user_input = st.text_area("Review:")

# Add a Predict button
if st.button("Predict"):
    if user_input:
        # Preprocess the input
        user_input_transformed = cv.transform([user_input]).toarray()

        # Predict the sentiment
        prediction = logreg_model.predict(user_input_transformed)
        
        # Map prediction to sentiment
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        # Display the result
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter a review to get a prediction.")
