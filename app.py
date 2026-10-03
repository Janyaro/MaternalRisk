from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MaternalRisk", page_icon="🤰", layout="wide")

BASE = Path(__file__).parent
FEAT = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
THRESH = 0.30   # chosen on training out-of-fold predictions, not on the test set


@st.cache_resource
def load_model():
    return joblib.load(BASE / "maternal_rf.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "maternal_clean.csv")


model, data = load_model(), load_data()
lo, hi = data[FEAT].min(), data[FEAT].max()

st.title("🤰 MaternalRisk")
st.caption("Maternal health risk classification (low / mid / high) from six vital signs")
st.warning("Research demo, not a medical device. The model was trained on a small dataset from "
           "rural Bangladesh and has not been validated on any other population.")

tab1, tab2, tab3 = st.tabs(["Risk check", "Model report", "Data quality"])

# ---------------- Tab 1 ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age (years)", int(lo.Age), int(hi.Age), 28)
    sbp = c1.slider("Systolic BP (mmHg)", int(lo.SystolicBP), int(hi.SystolicBP), 110)
    dbp = c2.slider("Diastolic BP (mmHg)", int(lo.DiastolicBP), int(hi.DiastolicBP), 75)
    bs = c2.slider("Blood sugar (mmol/L)", float(lo.BS), float(hi.BS), 7.5, step=0.1)
    temp = c3.slider("Body temperature (°F)", float(lo.BodyTemp), float(hi.BodyTemp), 98.0, step=0.1)
    hr = c3.slider("Heart rate (bpm)", int(lo.HeartRate), int(hi.HeartRate), 76)
    st.caption("Sliders are limited to the range seen in the training data.")

    row = pd.DataFrame([{"Age": age, "SystolicBP": sbp, "DiastolicBP": dbp,
                         "BS": bs, "BodyTemp": temp, "HeartRate": hr}])[FEAT]
    p = model.predict_proba(row)[0]
    probs = pd.DataFrame({"Risk level": ["Low", "Mid", "High"], "Probability": p})

    a, b = st.columns([1, 1])
    fig = px.bar(probs, x="Risk level", y="Probability", text=probs["Probability"].map("{:.0%}".format),
                 range_y=[0, 1], title="Predicted probabilities")
    fig.update_layout(margin=dict(t=50, b=10))
    a.plotly_chart(fig, width="stretch")
    b.metric("High-risk probability", f"{p[2]:.0%}")
    if p[2] >= THRESH:
        b.error(f"HIGH-RISK ALERT (probability ≥ {THRESH:.2f})")
    else:
        b.success(f"No high-risk alert (probability < {THRESH:.2f})")
    b.markdown(f"The alert threshold of **{THRESH:.2f}** trades extra false alarms for catching more "
               "high-risk cases. It was chosen on training data only.")
    b.caption("The mid-risk class is the weakest part of the model (see Model report), so treat "
              "the low/mid split with caution.")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Model comparison (training set, 5-fold CV repeated 3 times)")
    st.dataframe(pd.DataFrame({
        "Model": ["Dummy (baseline)", "Logistic Regression", "Random Forest (selected)", "Gradient Boosting"],
        "Accuracy": [0.517, 0.593, 0.640, 0.676],
        "Macro-F1": [0.227, 0.549, 0.599, 0.615],
        "High-risk recall": ["0.000", "0.640 ± 0.080", "0.718 ± 0.131", "0.685 ± 0.129"]}),
        width="stretch", hide_index=True)
    st.caption("Random Forest was selected beforehand for high-risk recall. "
               "Differences between the forest and boosting are within noise.")

    st.subheader("Final test set (91 rows, used once)")
    st.dataframe(pd.DataFrame({
        "Class": ["Low", "Mid", "High"], "Precision": [0.788, 0.333, 0.741],
        "Recall": [0.872, 0.190, 0.870], "Cases": [47, 21, 23]}),
        width="stretch", hide_index=True)
    st.write("Accuracy 0.714 | Macro-F1 0.624 (95% CI 0.53 to 0.72) | "
             "High-risk recall 0.870 (95% CI 0.71 to 1.00)")
    st.write(f"**High-risk alert at threshold {THRESH:.2f}:** 21 of 23 high-risk cases caught, "
             "11 false alarms. Recall 0.913 (CI 0.78 to 1.00), precision 0.656 (CI 0.48 to 0.82).")
    st.caption("With only 23 high-risk cases in the test set, these intervals are wide. "
               "The honest summary is high-risk recall of roughly 0.8 to 0.9.")

    a, b, c = st.columns(3)
    a.image(str(BASE / "cm.png"), caption="Confusion matrix (test set)")
    b.image(str(BASE / "importance.png"), caption="Permutation importance")
    c.image(str(BASE / "threshold.png"), caption="Threshold trade-off (training, out-of-fold)")
    st.markdown("**Error analysis.** Mid risk is the weak class (4 of 21 correct). Both high-risk cases "
                "missed at the alert threshold had a body temperature of 101 °F with normal blood sugar, "
                "while the model relies mostly on blood sugar and systolic blood pressure. "
                "This is two cases, so it is an observation, not a conclusion.")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("The dataset contains many duplicate rows")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows in CSV", "1,014")
    m2.metric("Duplicate rows", "562")
    m3.metric("Unique rows after cleaning", "451")
    st.markdown("Two rows with an impossible heart rate of 7 bpm were also removed. "
                "36 remaining rows have identical vital signs but different labels, so no model can be perfect.")
    acc = pd.DataFrame({"Setting": ["With duplicates", "Duplicates removed"], "Accuracy": [0.854, 0.641]})
    fig = px.bar(acc, x="Setting", y="Accuracy", text="Accuracy", range_y=[0, 1],
                 title="Random Forest accuracy (5-fold CV)")
    st.plotly_chart(fig, width="stretch")
    st.markdown("Duplicates let the same record appear in both the training and test folds, "
                "which inflates accuracy. This is why high accuracy figures reported for this dataset "
                "should be read with care.")
    cls = pd.DataFrame({"Class": ["Low", "Mid", "High"], "Before": [406, 336, 272], "After": [233, 106, 112]})
    st.dataframe(cls, width="stretch", hide_index=True)
