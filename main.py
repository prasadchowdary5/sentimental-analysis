import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load  # Correct import for joblib
import streamlit as st

# Define the paths to the dataset
pos_reviews_path = 'C:/Users/pchow/PycharmProjects/sentiment analysis model/pos'
neg_reviews_path = 'C:/Users/pchow/PycharmProjects/sentiment analysis model/neg'

# Function to load reviews
def load_reviews(path):
    reviews = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
    return reviews

# Load positive and negative reviews
pos_reviews = load_reviews(pos_reviews_path)
neg_reviews = load_reviews(neg_reviews_path)

# Create a DataFrame
df_pos = pd.DataFrame({'review': pos_reviews, 'sentiment': 1})
df_neg = pd.DataFrame({'review': neg_reviews, 'sentiment': 0})

# Combine positive and negative reviews
df = pd.concat([df_pos, df_neg], ignore_index=True)
print("Data successfully loaded and combined.")

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Building
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Model Evaluation
predictions = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Saving the Model and Vectorizer
dump(model, 'sentiment_model.pkl')
dump(vectorizer, 'vectorizer.pkl')

# Deployment using Streamlit
# To run this part, save the script and run `streamlit run script_name.py` in the terminal
if __name__ == "__main__":
    st.write("Real-Time Sentiment Analyzer")
    input_review = st.text_input("Enter Your Review...")

    if st.button("Analyze"):
        input_vectorized = vectorizer.transform([input_review])
        prediction = model.predict(input_vectorized)
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")

# Fine-Tuning and Optimization (not fully covered in this script)
# You can use GridSearchCV or RandomizedSearchCV for hyperparameter tuning and k-fold cross-validation
