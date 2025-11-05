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

# Accessible CSS: larger fonts, bigger inputs/buttons
st.markdown("""
<style>
html, body, [class*="css"]  {
  font-size: 18px !important;
}
button, .stButton>button {
  padding: 0.6rem 1rem;
  font-size: 18px;
}
input, select, textarea {
  font-size: 18px !important;
}
[data-testid="stMetricValue"] {
  font-size: 28px !important;
}
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

# Friendly UI labels -> model codes
STUDYTIME_UI = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]
STUDYTIME_MAP = {"<2 hours": 1, "2–5 hours": 2, "5–10 hours": 3, ">10 hours": 4}
FAILURES_UI = ["0 (none)", "1", "2", "3+"]
FAILURES_MAP = {"0 (none)": 0, "1": 1, "2": 2, "3+": 3}

# Initialize session settings once
if "settings" not in st.session_state:
    st.session_state.settings = {
        "easy_mode": True,              # hide advanced knobs by default
        "test_size": 0.2,
        "n_estimators": 300,
        "max_depth": 0,                 # 0 means "auto"
        "random_state": 42
    }


# =========================
# Helpers
# =========================
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
**Step 1 — Get a small CSV ready**  
If you don’t have one, click “Use built-in tiny sample” below.  
Make sure your CSV has a “**G3**” column (final grade), plus some of these: **G1, G2, age, absences, studytime, failures, sex, schoolsup, famsup, internet**.

**Step 2 — Go to the “Train” tab.**  
Upload your CSV. Press **Train model**. You’ll instantly see error metrics and a small preview of predictions.

**Step 3 — Predict**  
- **Predict (Single)** lets you type details for one student and see the predicted final grade.  
- **Predict (Batch)** lets you upload a CSV **without G3** and download predictions as a new CSV.
""")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Use built-in tiny sample"):
            st.session_state.demo_csv = small_sample_csv()
            st.success("Sample loaded. Go to the Train tab and click “Use demo CSV” there.")
    with c2:
        st.info("Tip: For best results, use at least ~100 rows and include G1 & G2.")


# -------------
# Train
# -------------
with tab_train:
    st.subheader("Train")
    st.markdown("Upload a CSV that **includes** the target column **G3** (final grade).")

    # UI choice: use built-in demo csv or upload
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

        # guardrails
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

            # Build pipeline from current settings
            num_cols, cat_cols = infer_feature_types(df)
            pre = build_preprocessor(num_cols, cat_cols)
            model = build_model_from_settings()
            pipe = Pipeline([("pre", pre), ("model", model)])

            if st.button("Train model", help="Click to train using current Training Settings"):
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
                st.session_state["trained_model"] = pipe
                st.success("Model trained and ready. Move to a Predict tab.")


# --------------------
# Predict (Single)
# --------------------
with tab_single:
    st.subheader("Predict (Single Student)")
    if "trained_model" not in st.session_state:
        st.info("Train a model first in the Train tab.")
    else:
        pipe = st.session_state["trained_model"]

        st.markdown("Fill in what you know. If you’re unsure, keep the default values.")
        c1, c2, c3 = st.columns(3)

        with c1:
            G1 = st.number_input("G1 — First period grade", min_value=0, max_value=20, value=10,
                                 help="0 to 20. Strong predictor.")
            G2 = st.number_input("G2 — Second period grade", min_value=0, max_value=20, value=10,
                                 help="0 to 20. Strong predictor.")
            absences = st.number_input("Absences — missed classes", min_value=0, max_value=200, value=5,
                                       help="How many classes missed in total.")

        with c2:
            age = st.number_input("Age (years)", min_value=10, max_value=30, value=17,
                                  help="Typical range 14–22 in this dataset.")
            studytime_label = st.selectbox("Weekly Study Time", STUDYTIME_UI, index=1,
                                           help="1=<2h, 2=2–5h, 3=5–10h, 4=>10h")
            failures_label = st.selectbox("Past Failures", FAILURES_UI, index=0,
                                          help="Number of past class failures.")

        with c3:
            sex = st.selectbox("Gender", ["F", "M"], index=0, help="Just F or M per dataset coding.")
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

        if st.button("Predict final grade (G3)"):
            try:
                pred = float(pipe.predict(df_row)[0])
                st.success(f"Predicted Final Grade (G3): {pred:.2f} / 20")
            except Exception as e:
                st.error(f"Prediction failed. Likely the model was trained on a very different schema. Details: {e}")


# --------------------
# Predict (Batch)
# --------------------
with tab_batch:
    st.subheader("Predict (Batch CSV)")
    if "trained_model" not in st.session_state:
        st.info("Train a model first in the Train tab.")
    else:
        pipe = st.session_state["trained_model"]
        st.markdown("Upload a CSV **without** G3. The app will add a **G3_pred** column and let you download it.")
        upb = st.file_uploader("Upload CSV for batch prediction (no G3 column)", type=["csv"])
        if upb:
            dfb = pd.read_csv(upb)
            dfb = clean_columns(dfb)
            st.dataframe(dfb.head(15), use_container_width=True)

            if TARGET in dfb.columns:
                st.warning("Remove G3 from this CSV before predicting.")
            if st.button("Run batch prediction"):
                try:
                    preds = pipe.predict(dfb)
                    out = dfb.copy()
                    out["G3_pred"] = preds
                    st.success("Done. Download the predictions below.")
                    st.download_button(
                        "Download predictions.csv",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv"
                    )
                except Exception as e:
                    st.error(f"Prediction failed. Likely schema mismatch with training data. Details: {e}")


# --------------------
# Training Settings
# --------------------
with tab_settings:
    st.subheader("Training Settings (Plain English)")
    st.markdown("""
These controls change **how the model learns**. If you prefer “just works,” leave **Easy Mode** on.

- **Easy Mode (recommended)**: sensible defaults for quick, stable training.
- **Test split**: how much data we **hold out** to check accuracy (0.2 = 20% for testing).
- **Number of trees**: more trees make results steadier but can be a bit slower.
- **Max depth**: how detailed each tree can get (0 = let the model decide). Very big depth can overfit.
- **Random seed**: keeps results repeatable.
""")

    s = st.session_state.settings
    s["easy_mode"] = st.checkbox("Easy Mode (auto-tuned defaults)", value=s["easy_mode"])

    if s["easy_mode"]:
        st.info("Easy Mode is ON. We’ll use: test split 0.2, 300 trees, auto depth, random seed 42.")
        left, right = st.columns(2)
        with left:
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
            s["test_size"] = st.slider("Test split (fraction kept for checking accuracy)", 0.1, 0.4, float(s["test_size"]), 0.05)
            s["n_estimators"] = st.slider("Number of trees (stability vs speed)", 50, 1000, int(s["n_estimators"]), 50)
        with c2:
            s["max_depth"] = st.slider("Max depth (0 = auto, higher = more detail)", 0, 50, int(s["max_depth"]), 1)
            s["random_state"] = st.number_input("Random seed (keeps results repeatable)", value=int(s["random_state"]), step=1)

        if st.button("Save settings"):
            st.success("Settings saved. Go to Train and run again.")


# --------------------
# Help & Full Forms
# --------------------
with tab_help:
    st.subheader("Full Forms / What fields mean")
    st.markdown("""
**G1** — First period grade (0–20)  
**G2** — Second period grade (0–20)  
**G3** — Final grade (0–20, what we predict)  
**age** — Age in years  
**absences** — Number of missed classes  
**studytime** — Weekly study time (1=<2h, 2=2–5h, 3=5–10h, 4=>10h)  
**failures** — Past class failures (0 means none)  
**sex** — Student gender (M/F)  
**schoolsup** — Extra educational support at school (yes/no)  
**famsup** — Family educational support (yes/no)  
**internet** — Internet access at home (yes/no)
""")
    st.markdown("""
**Tips for better accuracy**
- Include **G1 and G2**. They are powerful predictors.
- Use at least **100+ rows** if possible.
- Keep one subject per model (don’t mix Maths & Portuguese data for the same model until you know what you’re doing).
""")
