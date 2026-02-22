import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("ML Practical 3 – Spam Email Classifier (SVM)")

emails = [
    "Congratulations! You've won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don't forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday's discussion",
    "Please confirm your attendance for the workshop",
    "Let's finalize the budget proposal by Friday",
]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"**Improved Model Accuracy:** {accuracy:.4f}")

st.subheader("Try the Classifier")
new_email = st.text_input("Enter a new email message:", placeholder="Type an email message here...")

if new_email:
    new_email_vectorized = vectorizer.transform([new_email])
    prediction = svm_model.predict(new_email_vectorized)

    if prediction[0] == 1:
        st.error("Result: The email is **spam**.")
    else:
        st.success("Result: The email is **not spam**.")