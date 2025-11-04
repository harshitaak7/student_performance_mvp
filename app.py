import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ---------- Page setup ----------
st.set_page_config(page_title="Student Performance — MVP Prototype", layout="wide")
st.title("Student Performance — MVP Prototype")
st.caption("Upload CSV → Train → Evaluate → Predict (single/batch). Minimal, but real.")

# ---------- Field glossary / full forms ----------
with st.expander("Full forms / what each field means"):
    st.markdown("""
**G1** — First period grade (0–20)  
**G2** — Second period grade (0–20)  
**G3** — Final grade (0–20, target we predict)  
**age** — Age in years  
**absences** — Number of missed classes  
**studytime** — Weekly study time  
&nbsp;&nbsp;&nbsp;&nbsp;1 = <2 hours, 2 = 2–5 hours, 3 = 5–10 hours, 4 = >10 hours  
**failures** — Past class failures (0 = none)  
**sex** — Student gender (M/F)  
**schoolsup** — Extra educational support at school (yes/no)  
**famsup** — Family educational support (yes/no)  
**internet** — Internet access at home (yes/no)
""")

# ---------- MVP helpers ----------
TARGET = "G3"

NUMERIC_DEFAULT = ["age", "absences", "G1", "G2"]
CATEG_DEFAULT = ["sex", "studytime", "failures", "schoolsup", "famsup", "internet"]

STUDYTIME_UI = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]
STUDYTIME_MAP = {"<2 hours": 1, "2–5 hours": 2, "5–10 hours": 3, ">10 hours": 4}

def clean_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_feature_types(df: pd.DataFrame):
    # start with sensible defaults if present
    num = [c for c in NUMERIC_DEFAULT if c in df.columns and c != TARGET]
    cat = [c for c in CATEG_DEFAULT if c in df.columns and c != TARGET]
    # add any extra numeric columns (except target)
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in num and c != TARGET:
            num.append(c)
    # add any extra object/bool/category columns
    for c in df.select_dtypes(include=["object", "bool", "category"]).columns:
        if c not in cat and c != TARGET:
            cat.append(c)
    # de-dup, preserve order
    seen = set(); num = [x for x in num if not (x in seen or seen.add(x))]
    seen = set(); cat = [x for x in cat if not (x in seen or seen.add(x))]
    return num, cat

def build_preprocessor(num_cols, cat_cols):
    num_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols)
    ], remainder="drop")
    return pre

def split_xy(df: pd.DataFrame, target=TARGET):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' missing in CSV.")
    X = df.drop(columns=[target])
    y = df[target].astype(float)
    return X, y

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Training Settings")
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RandomForest: n_estimators", 50, 800, 300, 50)
    max_depth = st.slider("RandomForest: max_depth (0 = auto)", 0, 40, 0, 1)
    random_state = st.number_input("random_state", value=42, step=1)

# ---------- Tabs ----------
tab_train, tab_single, tab_batch = st.tabs(["1) Train", "2) Predict (single)", "3) Predict (batch)"])

# ---------- 1) Train ----------
with tab_train:
    st.subheader("Upload training data (CSV must include G3 target)")
    st.caption("Tip: start with a small sample CSV to validate the pipeline before using a large dataset.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="train_csv")

    if up:
        df = pd.read_csv(up)
        df = clean_columns(df)
        st.dataframe(df.head(), use_container_width=True)

        if TARGET not in df.columns:
            st.error(f"CSV must include target column '{TARGET}'.")
        else:
            critical_missing = [c for c in ["G1", "G2"] if c not in df.columns]
            if critical_missing:
                st.warning(f"Your data is missing {critical_missing}. Expect weaker predictions; these are strong predictors.")

            try:
                X, y = split_xy(df)
            except Exception as e:
                st.error(str(e))
                st.stop()

            num_cols, cat_cols = infer_feature_types(df)
            pre = build_preprocessor(num_cols, cat_cols)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=None if max_depth == 0 else max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            pipe = Pipeline([("pre", pre), ("model", model)])

            if st.button("Train model"):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                pipe.fit(X_tr, y_tr)
                y_pr = pipe.predict(X_te)

                mae = mean_absolute_error(y_te, y_pr)
                rmse = np.sqrt(mean_squared_error(y_te, y_pr))
                r2 = r2_score(y_te, y_pr)

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.2f}")
                c2.metric("RMSE", f"{rmse:.2f}")
                c3.metric("R²", f"{r2:.2f}")

                # show a small preview of predictions vs truth
                preview = pd.DataFrame({
                    "y_true": y_te.reset_index(drop=True),
                    "y_pred": pd.Series(y_pr)
                })
                st.dataframe(preview.head(12), use_container_width=True)

                # keep the trained model in session for the Predict tabs
                st.session_state["trained_model"] = pipe
                st.success("Model trained and kept in session. Go to the Predict tabs.")

# ---------- 2) Single prediction ----------
with tab_single:
    st.subheader("Single student prediction")
    if "trained_model" not in st.session_state:
        st.info("Train a model first in tab 1.")
    else:
        pipe = st.session_state["trained_model"]

        col1, col2, col3 = st.columns(3)

        with col1:
            G1 = st.number_input("G1 — First period grade (0–20)", min_value=0, max_value=20, value=10)
            G2 = st.number_input("G2 — Second period grade (0–20)", min_value=0, max_value=20, value=10)
            absences = st.number_input("Absences — missed classes", min_value=0, max_value=200, value=5)

        with col2:
            age = st.number_input("Age (years)", min_value=10, max_value=30, value=17)
            studytime_label = st.selectbox("Weekly Study Time", STUDYTIME_UI, index=1)
            failures_label = st.selectbox("Past Failures", ["0 (none)", "1", "2", "3+"], index=0)

        with col3:
            sex = st.selectbox("Gender", ["F", "M"], index=0)
            schoolsup_label = st.selectbox("School Support (extra help)", ["Yes", "No"], index=1)
            famsup_label = st.selectbox("Family Support", ["Yes", "No"], index=1)
            internet_label = st.selectbox("Internet at home", ["Yes", "No"], index=0)

        # map UI labels back to model-friendly encodings
        failures_map = {"0 (none)": 0, "1": 1, "2": 2, "3+": 3}
        row = {
            "sex": sex,
            "age": age,
            "studytime": STUDYTIME_MAP[studytime_label],
            "failures": failures_map[failures_label],
            "schoolsup": schoolsup_label.lower(),
            "famsup": famsup_label.lower(),
            "internet": internet_label.lower(),
            "absences": absences,
            "G1": G1,
            "G2": G2
        }
        df_row = pd.DataFrame([row])

        if st.button("Predict G3"):
            pred = float(pipe.predict(df_row)[0])
            st.success(f"Predicted Final Grade (G3): {pred:.2f} / 20")

# ---------- 3) Batch prediction ----------
with tab_batch:
    st.subheader("Batch prediction (CSV without G3)")
    if "trained_model" not in st.session_state:
        st.info("Train a model first in tab 1.")
    else:
        pipe = st.session_state["trained_model"]
        upb = st.file_uploader("Upload CSV for batch inference (no G3 column)", type=["csv"], key="batch_csv")
        if upb:
            dfb = pd.read_csv(upb)
            dfb = clean_columns(dfb)
            st.dataframe(dfb.head(), use_container_width=True)

            if TARGET in dfb.columns:
                st.warning("Remove G3 from this CSV; batch prediction expects unknown final grades.")

            if st.button("Run batch prediction"):
                try:
                    preds = pipe.predict(dfb)
                except Exception as e:
                    st.error(f"Prediction failed. Likely schema mismatch with training data. Details: {e}")
                    st.stop()

                out = dfb.copy()
                out["G3_pred"] = preds
                st.success("Done. Download predictions below.")
                st.download_button(
                    "Download predictions.csv",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv"
                )
