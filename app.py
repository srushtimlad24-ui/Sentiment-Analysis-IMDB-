import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("IMDB Sentiment Analysis")

uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

if uploaded_file is not None:
    
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Convert sentiment labels to numeric
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    st.write(df.head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'],
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes model
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = nb.predict(X_test_tfidf)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='mako',
        ax=ax
    )
    ax.set_title("Sentiment Analysis Confusion Matrix")
    st.pyplot(fig)

    # Test custom review
    user_review = st.text_input("Test a Review")

    if user_review:
        vec = tfidf.transform([user_review])
        prediction = nb.predict(vec)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
