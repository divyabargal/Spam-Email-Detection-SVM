
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the Kaggle dataset
df = pd.read_csv("spam_ham_dataset.csv")

# 2. Keep the required columns and remove missing records
df = df[["text", "label"]].dropna()

# Remove empty email text
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# 3. Separate input text and target labels
X = df["text"]
y = df["label"].str.strip().str.lower()

print("Dataset size:", len(df))
print("\nClass distribution:")
print(y.value_counts())

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Create TF-IDF + SVM pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=20000
        )
    ),
    (
        "svm",
        LinearSVC(random_state=42)
    )
])

# 6. Train the model
print("\nTraining the SVM model...")
model.fit(X_train, y_train)

# 7. Evaluate on unseen test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        zero_division=0
    )
)

print("Confusion Matrix (rows=true, columns=predicted):")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )
)

# 8. Save the trained model
joblib.dump(model, "spam_svm_model.joblib")

print("\nModel saved as spam_svm_model.joblib")
print("Training completed successfully!")
