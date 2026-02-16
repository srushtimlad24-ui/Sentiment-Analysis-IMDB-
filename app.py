# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("IMDB Sentiment Analysis - Naive Bayes")

# ---------------------------------
# Upload Dataset
# ---------------------------------

uploaded_file = st.file_uploader("Upload IMDB_Dataset.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, encoding='latin-1')

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------
    # Preprocessing
    # ---------------------------------

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # ---------------------------------
    # Model Training
    # ---------------------------------

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    y_pred = nb_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    # ---------------------------------
    # Results
    # ---------------------------------

    st.subheader("Model Performance")

    st.write(f"Accuracy: {accuracy:.4f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ---------------------------------
    # Confusion Matrix
    # ---------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    for (i, j), val in pd.DataFrame(cm).stack().items():
        ax.text(j, i, int(val), ha='center', va='center')

    ax.set_xticklabels(['', 'Negative', 'Positive'])
    ax.set_yticklabels(['', 'Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    st.pyplot(fig)

    # ---------------------------------
    # Custom Prediction
    # ---------------------------------

    st.subheader("Test Custom Review")

    user_review = st.text_area("Enter a movie review:")

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            review_tfidf = tfidf.transform([user_review])
            prediction = nb_model.predict(review_tfidf)

            if prediction[0] == 1:
                st.success("Sentiment: Positive")
            else:
                st.error("Sentiment: Negative")

else:
    st.info("Please upload the IMDB_Dataset.csv file to continue.")
