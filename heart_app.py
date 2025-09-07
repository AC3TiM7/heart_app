import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================================
# Streamlit App
# ================================
st.title("Heart Disease Prediction App ❤️")

# Fixed file path
file_path = r"C:\Users\user\OneDrive\Homeworks\AI\cleaned_heart.xlsx"

# Valid ranges for each feature
valid_ranges = {
    "Age": (0.00, 1.00, "float"),
    "Sex": ([0, 1], "int"),
    "ChestPainType": ([0, 1, 2, 3], "int"),
    "RestingBP": (0.0000, 1.0000, "float"),
    "Cholesterol": (0.0000, 1.0000, "float"),
    "FastingBS": ([0, 1], "int"),
    "RestingECG": ([0, 1, 2], "int"),
    "MaxHR": (0.0000, 1.0000, "float"),
    "ExerciseAngina": ([0, 1], "int"),
    "Oldpeak": (0.0000, 1.0000, "float"),
    "ST_Slope": ([0, 1, 2], "int"),
    "HeartDisease": ([0, 1], "int"),
}

try:
    # Read dataset
    df = pd.read_excel(file_path)
    st.write("### Preview of Data", df.head())

    # Features (all except last col) & Target (last col)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    valid_cols = [col for col in X.columns if col in valid_ranges]

    # Train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================================
    # Train ALL Models
    # ================================
    models = {
        "SVM": SVC(kernel="linear", C=1, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name in ["SVM", "KNN"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:  # Random Forest (not scaled)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    # Display metrics for all models
    st.write("## Model Comparison Results")
    results_df = pd.DataFrame([
        [name, res["accuracy"], res["precision"], res["recall"], res["f1"]]
        for name, res in results.items()
    ], columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
    st.dataframe(results_df, use_container_width=True)

    # ================================
    # Prediction demo
    # ================================
    st.write("### Try a Prediction")

    if "user_input" not in st.session_state:
        st.session_state.user_input = [None] * len(valid_cols)

    # Random generator
    if st.button("Generate Random Data"):
        random_data = []
        for col in valid_cols:
            vr = valid_ranges[col]
            if isinstance(vr[0], list):  # discrete
                val = np.random.choice(vr[0])
            else:  # continuous
                val = round(np.random.uniform(vr[0], vr[1]), 2) if vr[-1] == "float" else int(
                    np.random.uniform(vr[0], vr[1])
                )
            random_data.append(val)
        st.session_state.user_input = random_data

    # ================================
    # Input descriptions
    # ================================
    input_descriptions = {
        "Age": " Age of the patient (normalized between 0 and 1).",
        "Sex": " 0 = Male, 1 = Female.",
        "ChestPainType": " 0 = Asymptomatic, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Typical Angina.",
        "RestingBP": " Resting blood pressure (normalized between 0 and 1).",
        "Cholesterol": " Serum cholesterol in mg/dl (normalized between 0 and 1).",
        "FastingBS": " Fasting blood sugar (0 = < 120 mg/dl, 1 = > 120 mg/dl).",
        "RestingECG": " 0 = LVH by Estes' criteria, 1 = Normal, 2 = ST-T wave abnormality.",
        "MaxHR": " Maximum heart rate achieved (normalized between 0 and 1).",
        "ExerciseAngina": " 0 = No, 1 = Yes.",
        "Oldpeak": " ST depression induced by exercise relative to rest (normalized between 0 and 1).",
        "ST_Slope": " 0 = Downsloping, 1 = Flat, 2 = Upsloping."
    }

    # ================================
    # Show inputs (with descriptions)
    # ================================
    inputs = []
    for i, col in enumerate(valid_cols):
        vr = valid_ranges[col]
        current_val = st.session_state.user_input[i]

        if isinstance(vr[0], list):
            val = st.selectbox(
                f"Enter {col}", vr[0],
                index=vr[0].index(current_val) if current_val in vr[0] else 0,
                key=f"input_{col}"
            )
        else:
            default_val = float((vr[0] + vr[1]) / 2) if current_val is None else float(current_val)
            val = st.number_input(
                f"Enter {col}",
                min_value=float(vr[0]),
                max_value=float(vr[1]),
                value=default_val,
                format="%.4f",
                key=f"input_{col}"
            )

        # ✅ Add description under each input
        if col in input_descriptions:
            st.caption(input_descriptions[col])

        inputs.append(val)



    # Update session state with edited values
    st.session_state.user_input = inputs

    # Prediction button
    if st.button("Predict"):
        valid = True
        for i, col in enumerate(valid_cols):
            vr = valid_ranges[col]
            val = st.session_state.user_input[i]
            if isinstance(vr[0], list):
                if val not in vr[0]:
                    valid = False
            else:
                if not (vr[0] <= val <= vr[1]):
                    valid = False

        if not valid:
            st.error("❌ Invalid input format")
        else:
            st.write("### Predictions from All Models")
            for name, res in results.items():
                if name in ["SVM", "KNN"]:
                    user_data = scaler.transform([st.session_state.user_input])
                else:  # Random Forest
                    user_data = [st.session_state.user_input]

                prediction = res["model"].predict(user_data)[0]
                st.success(f"{name}: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")

except Exception as e:
    st.error(f"Error loading file: {e}")
