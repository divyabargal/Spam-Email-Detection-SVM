
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "spam_svm_model.joblib"
DATA_PATH = BASE_DIR / "spam_ham_dataset.csv"

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]
    return df


@st.cache_data
def calculate_metrics():
    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    model = load_model()
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        ),
        "matrix": confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        )
    }


# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.title("📧 Spam Email Detection System")
st.write(
    "Machine learning-based email classification using "
    "TF-IDF and Support Vector Machine (SVM)."
)

st.divider()

# Dashboard
df = load_dataset()
metrics = calculate_metrics()

total = len(df)
spam_count = int((df["label"] == "spam").sum())
ham_count = int((df["label"] == "ham").sum())

st.subheader("📊 Project Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Total Emails", f"{total:,}")
c2.metric("Spam Emails", f"{spam_count:,}")
c3.metric("Legitimate Emails", f"{ham_count:,}")

st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
m2.metric("Precision (Spam)", f"{metrics['precision'] * 100:.2f}%")
m3.metric("Recall (Spam)", f"{metrics['recall'] * 100:.2f}%")
m4.metric("F1-Score (Spam)", f"{metrics['f1'] * 100:.2f}%")

with st.expander("View Confusion Matrix"):
    matrix = pd.DataFrame(
        metrics["matrix"],
        index=["Actual Ham", "Actual Spam"],
        columns=["Predicted Ham", "Predicted Spam"]
    )
    st.dataframe(matrix, use_container_width=True)

st.divider()

# Email prediction
st.subheader("🔍 Check an Email")

email_text = st.text_area(
    "Enter email content",
    placeholder="Paste the email message here...",
    height=180,
    key="email_text"
)

c1, c2 = st.columns(2)

with c1:
    check = st.button(
        "Check Email",
        type="primary",
        use_container_width=True
    )

def clear_email():
    st.session_state.email_text = ""

with c2:
    st.button(
        "Clear",
        use_container_width=True,
        on_click=clear_email
    )


if check:
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        model = load_model()
        prediction = model.predict([email_text])[0]

        if prediction == "spam":
            st.error("🚨 SPAM EMAIL DETECTED")
            st.write("The model classified this message as spam.")
        else:
            st.success("✅ LEGITIMATE EMAIL")
            st.write("The model classified this message as legitimate.")

        st.session_state.history.insert(
            0,
            {
                "Email Preview": email_text[:100],
                "Prediction": prediction.upper()
            }
        )

st.divider()

# Prediction history
st.subheader("🕘 Recent Predictions")

if st.session_state.history:
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Your prediction history will appear here.")

st.divider()

st.subheader("ℹ️ About the Project")
st.write(
    "Dataset: Kaggle Spam Mails Dataset. "
    "TF-IDF converts email text into numerical features, "
    "and a trained linear SVM classifier predicts spam or ham."
)

st.caption(
    "Academic Mini Project | Python | Scikit-learn | Streamlit"
)
