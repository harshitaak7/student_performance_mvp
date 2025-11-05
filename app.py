import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# =========================
# Page setup + Accessibility
# =========================
st.set_page_config(page_title="Student Performance — Simple App", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 18px !important; }
button, .stButton>button { padding: 0.6rem 1rem; font-size: 18px; }
input, select, textarea { font-size: 18px !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

st.title("Student Performance — Simple App")
st.caption("Upload → Train → Evaluate → Predict. Designed to be easy for any age. No jargon needed.")

# =========================
# Globals / Defaults
# =========================
TARGET = "G3"
NUMERIC_DEFAULT = ["age", "absences", "G1", "G2"]
CATEG_DEFAULT = ["sex", "studytime", "failures", "schoolsup", "famsup", "internet"]

STUDYTIME_UI = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]
STUDYTIME_MAP = {"<2 hours": 1, "2–5 hours": 2, "5–10 hours": 3, ">10 hours": 4}
FAILURES_UI = ["0 (none)", "1", "2", "3+"]
FAILURES_MAP = {"0 (none)": 0, "1": 1, "2": 2, "3+": 3}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "easy_mode": True,
        "test_size": 0.2,
        "n_estimators": 300,
        "max_depth": 0,
        "random_state": 42
    }

# Will store training artifacts for schema alignment
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "train_num_cols" not in st.session_state:
    st.session_state.train_num_cols = None
if "train_cat_cols" not in st.session_state:
    st.session_state.train_cat_cols = None
if "train_required_cols" not in st.session_state:
    st.session_state.train_required_cols = None  # union, ordered as in X during training

# =========================
# Helpers
# =========================
def clean_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_feature_types(df: pd.DataFrame):
    num = [c for c in NUMERIC_DEFAULT if c in df.columns and c != TARGET]
    cat = [c for c in CATEG_DEFAULT if c in df.columns and c != TARGET]
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in num and c != TARGET:
            num.append(c)
    for c in df.select_dtypes(include=["object", "bool", "category"]).columns:
        if c not in cat and c != TARGET:
            cat.append(c)
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
    return ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols)
    ], remainder="drop")

def split_xy(df: pd.DataFrame, target=TARGET):
    if target not in df.columns:
        raise ValueError(f"CSV must include '{target}' (the final grade).")
    X = df.drop(columns=[target])
    y = df[target].astype(float)
    return X, y

def build_model_from_settings():
    s = st.session_state.settings
    return RandomForestRegressor(
        n_estimators=int(s["n_estimators"]),
        max_depth=None if int(s["max_depth"]) == 0 else int(s["max_depth"]),
        random_state=int(s["random_state"]),
        n_jobs=-1
    )

def small_sample_csv() -> str:
    return (
        "sex,age,studytime,failures,schoolsup,famsup,internet,absences,G1,G2,G3\n"
        "F,17,2,0,no,yes,yes,4,10,11,12\n"
        "M,18,1,1,yes,no,no,10,8,9,9\n"
        "F,16,3,0,no,yes,yes,2,13,14,15\n"
        "M,17,2,2,no,yes,yes,12,9,10,10\n"
        "F,16,2,0,no,no,no,6,11,12,13\n"
    )

def align_to_training_schema(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Make incoming prediction DataFrame match training schema:
    - add any missing required columns as NaN (imputer will handle)
    - drop extra columns
    - reorder columns to match training X
    """
    req = st.session_state.train_required_cols
    if not req:
        raise RuntimeError("Training schema not found in session. Train a model first.")

    df = df_in.copy()
    # add missing columns as NaN
    for col in req:
        if col not in df.columns:
            df[col] = np.nan

    # drop extras and reorder
    df = df[req]
    return df

def make_template_csv() -> bytes:
    """
    Create a CSV header-only template that matches the training columns (no G3).
    """
    req = st.session_state.train_required_cols
    if not req:
        return b""
    # header-only CSV
    buf = ",".join(req) + "\n"
    return buf.encode("utf-8")

# =========================
# Tabs
# =========================
tab_home, tab_train, tab_single, tab_batch, tab_settings, tab_help = st.tabs(
    ["Start", "Train", "Predict (Single)", "Predict (Batch)", "Training Settings", "Help & Full Forms"]
)

# -------------
# Start (guided)
# -------------
with tab_home:
    st.subheader("Quick Start")
    st.markdown("""
**Step 1 — Get a CSV**  
If you don’t have one, click “Use built-in tiny sample.”  
Include **G3** to train. Include **G1 & G2** for better accuracy.

**Step 2 — Train**  
Upload in the **Train** tab → click **Train model** → see metrics.

**Step 3 — Predict**  
Use **Single** or **Batch**. For Batch, either:
- Upload the **template** (download it in Train tab) filled with your students, or
- Upload a CSV with exactly the **same columns** used during training (any order is fine).
""")
    if st.button("Use built-in tiny sample"):
        st.session_state.demo_csv = small_sample_csv()
        st.success("Sample loaded. Go to Train tab → tick 'Use demo CSV'.")

# -------------
# Train
# -------------
with tab_train:
    st.subheader("Train")
    st.markdown("Upload a CSV that **includes** the target column **G3** (final grade).")

    use_demo = st.checkbox("Use demo CSV (tiny built-in sample)", value=("demo_csv" in st.session_state))
    up = None
    if use_demo:
        if "demo_csv" not in st.session_state:
            st.session_state.demo_csv = small_sample_csv()
        up = StringIO(st.session_state.demo_csv)
    else:
        up = st.file_uploader("Upload CSV", type=["csv"], help="CSV must include column G3 for training")

    if up:
        df = pd.read_csv(up)
        df = clean_columns(df)
        st.dataframe(df.head(15), use_container_width=True)

        if TARGET not in df.columns:
            st.error(f"Your CSV is missing '{TARGET}'. Add it to train.")
        else:
            critical_missing = [c for c in ["G1", "G2"] if c not in df.columns]
            if critical_missing:
                st.warning(f"Missing strong predictors {critical_missing}. Accuracy will likely be weaker.")

            try:
                X, y = split_xy(df)
            except Exception as e:
                st.error(str(e))
                st.stop()

            # Build pipeline and remember schema
            num_cols, cat_cols = infer_feature_types(df)
            pre = build_preprocessor(num_cols, cat_cols)
            model = build_model_from_settings()
            pipe = Pipeline([("pre", pre), ("model", model)])

            if st.button("Train model", help="Trains using current Training Settings"):
                s = st.session_state.settings
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=float(s["test_size"]), random_state=int(s["random_state"])
                )
                pipe.fit(X_tr, y_tr)
                y_pr = pipe.predict(X_te)

                mae = mean_absolute_error(y_te, y_pr)
                rmse = np.sqrt(mean_squared_error(y_te, y_pr))
                r2 = r2_score(y_te, y_pr)

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{mae:.2f}")
                m2.metric("RMSE", f"{rmse:.2f}")
                m3.metric("R²", f"{r2:.2f}")

                preview = pd.DataFrame({
                    "Actual (G3)": y_te.reset_index(drop=True),
                    "Predicted (G3)": pd.Series(y_pr)
                })
                st.markdown("**Prediction preview (sample):**")
                st.dataframe(preview.head(12), use_container_width=True)

                # Save to session for prediction tabs
                st.session_state.trained_model = pipe
                # Save training schema: the columns present in X (order matters for ColumnTransformer by name)
                st.session_state.train_num_cols = num_cols
                st.session_state.train_cat_cols = cat_cols
                st.session_state.train_required_cols = list(X.columns)  # <- critical for alignment

                st.success("Model trained and ready. Move to a Predict tab.")

                # Offer a header-only template matching training schema (no G3)
                if st.session_state.train_required_cols:
                    st.download_button(
                        "Download prediction template (CSV header)",
                        data=make_template_csv(),
                        file_name="prediction_template.csv",
                        mime="text/csv",
                        help="Use this template for Batch Predict. Fill rows and upload."
                    )

# --------------------
# Predict (Single)
# --------------------
with tab_single:
    st.subheader("Predict (Single Student)")
    if st.session_state.trained_model is None:
        st.info("Train a model first in the Train tab.")
    else:
        pipe = st.session_state.trained_model

        st.markdown("Fill what you know. Defaults are fine if you’re unsure.")
        c1, c2, c3 = st.columns(3)

        with c1:
            G1 = st.number_input("G1 — First period grade", min_value=0, max_value=20, value=10,
                                 help="0–20. Strong predictor.")
            G2 = st.number_input("G2 — Second period grade", min_value=0, max_value=20, value=10,
                                 help="0–20. Strong predictor.")
            absences = st.number_input("Absences — missed classes", min_value=0, max_value=200, value=5)

        with c2:
            age = st.number_input("Age (years)", min_value=10, max_value=30, value=17,
                                  help="Typical range 14–22 in this dataset.")
            studytime_label = st.selectbox("Weekly Study Time", STUDYTIME_UI, index=1,
                                           help="1=<2h, 2=2–5h, 3=5–10h, 4=>10h")
            failures_label = st.selectbox("Past Failures", FAILURES_UI, index=0,
                                          help="Number of past class failures.")

        with c3:
            sex = st.selectbox("Gender", ["F", "M"], index=0, help="F or M per dataset coding.")
            schoolsup_label = st.selectbox("School Support (extra help)", ["Yes", "No"], index=1)
            famsup_label = st.selectbox("Family Support", ["Yes", "No"], index=1)
            internet_label = st.selectbox("Internet at home", ["Yes", "No"], index=0)

        row = {
            "sex": sex,
            "age": age,
            "studytime": STUDYTIME_MAP[studytime_label],
            "failures": FAILURES_MAP[failures_label],
            "schoolsup": schoolsup_label.lower(),
            "famsup": famsup_label.lower(),
            "internet": internet_label.lower(),
            "absences": absences,
            "G1": G1,
            "G2": G2
        }
        df_row = pd.DataFrame([row])

        # Align single-row input to training schema before predict
        try:
            df_row_aligned = align_to_training_schema(df_row)
        except Exception as e:
            st.error(str(e))
            st.stop()

        if st.button("Predict final grade (G3)"):
            try:
                pred = float(pipe.predict(df_row_aligned)[0])
                st.success(f"Predicted Final Grade (G3): {pred:.2f} / 20")
            except Exception as e:
                st.error(f"Prediction failed. Details: {e}")

# --------------------
# Predict (Batch)
# --------------------
with tab_batch:
    st.subheader("Predict (Batch CSV)")
    if st.session_state.trained_model is None:
        st.info("Train a model first in the Train tab.")
    else:
        pipe = st.session_state.trained_model
        st.markdown("Upload a CSV **without** G3. We’ll align columns automatically to match training.")
        upb = st.file_uploader("Upload CSV for batch prediction (no G3 column)", type=["csv"])
        if upb:
            dfb = pd.read_csv(upb)
            dfb = clean_columns(dfb)
            st.dataframe(dfb.head(15), use_container_width=True)

            if TARGET in dfb.columns:
                st.warning("Remove G3 from this CSV before predicting.")

            if st.button("Run batch prediction"):
                try:
                    dfb_aligned = align_to_training_schema(dfb)
                    preds = pipe.predict(dfb_aligned)
                    out = dfb.copy()
                    out["G3_pred"] = preds
                    st.success("Done. Download the predictions below.")
                    st.download_button(
                        "Download predictions.csv",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv"
                    )
                except Exception as e:
                    st.error(f"Prediction failed. Details: {e}")

# --------------------
# Training Settings
# --------------------
with tab_settings:
    st.subheader("Training Settings (Plain English)")
    st.markdown("""
These change **how the model learns**. If unsure, leave **Easy Mode** on.

- **Easy Mode**: good defaults.
- **Test split**: fraction held out to check accuracy (0.2 = 20%).
- **Number of trees**: more = steadier results, a bit slower.
- **Max depth**: how detailed each tree is (0 = auto). Very large can overfit.
- **Random seed**: keeps results repeatable.
""")

    s = st.session_state.settings
    s["easy_mode"] = st.checkbox("Easy Mode (auto defaults)", value=s["easy_mode"])

    if s["easy_mode"]:
        st.info("Using: test split 0.2, 300 trees, auto depth, seed 42.")
        if st.button("Apply defaults now"):
            st.session_state.settings.update({
                "test_size": 0.2,
                "n_estimators": 300,
                "max_depth": 0,
                "random_state": 42
            })
            st.success("Defaults applied.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            s["test_size"] = st.slider("Test split", 0.1, 0.4, float(s["test_size"]), 0.05)
            s["n_estimators"] = st.slider("Number of trees", 50, 1000, int(s["n_estimators"]), 50)
        with c2:
            s["max_depth"] = st.slider("Max depth (0 = auto)", 0, 50, int(s["max_depth"]), 1)
            s["random_state"] = st.number_input("Random seed", value=int(s["random_state"]), step=1)

        if st.button("Save settings"):
            st.success("Settings saved. Train again to apply.")

# --------------------
# Help & Full Forms
# --------------------
with tab_help:
    st.subheader("Full Forms / Field meanings")
    st.markdown("""
**G1** — First period grade (0–20)  
**G2** — Second period grade (0–20)  
**G3** — Final grade (0–20, what we predict)  
**age** — Age in years  
**absences** — Number of missed classes  
**studytime** — Weekly study time (1=<2h, 2=2–5h, 3=5–10h, 4=>10h)  
**failures** — Past class failures (0 means none)  
**sex** — Student gender (M/F)  
**schoolsup** — Extra school support (yes/no)  
**famsup** — Family educational support (yes/no)  
**internet** — Internet access at home (yes/no)
""")
    st.markdown("""
**Tips for better accuracy**
- Include **G1 and G2**.
- Use **100+ rows** if possible.
- Keep one subject per model (don’t mix Math & Portuguese in the same training run initially).
""")
