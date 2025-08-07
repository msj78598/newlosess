import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# تحميل نموذج Isolation Forest
@st.cache_resource
def load_model():
    return joblib.load("model/isolation_forest.pkl")

# تحليل الشذوذ
def analyze_anomalies(df, model):
    feature_cols = ['V1', 'V2', 'V3', 'A1', 'A2', 'A3']
    df.columns = df.columns.str.strip()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso_scores = model.decision_function(X_scaled)
    iso_preds = model.predict(X_scaled)

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    lof_preds = lof.fit_predict(X_scaled)

    # تجميع النتائج
    df_result = df.copy()
    df_result['IForest'] = iso_preds
    df_result['LOF'] = lof_preds

    # تصنيف الحالات
    def classify(row):
        if row['IForest'] == -1 and row['LOF'] == -1:
            return 'أولوية 1 - كلا النموذجين'
        elif row['IForest'] == -1 or row['LOF'] == -1:
            return 'أولوية 2 - نموذج واحد'
        else:
            return 'سليم'

    df_result['التصنيف'] = df_result.apply(classify, axis=1)
    return df_result

# إعداد الصفحة
st.set_page_config(page_title="تحليل فاقد الطاقة", layout="wide")
st.title("⚡ نظام اكتشاف حالات الفاقد باستخدام الذكاء الاصطناعي")
st.markdown("📊 لتحليل البيانات وتحديد الحالات الشاذة (CSV أو Excel) يرجى رفع ملف الأحمال")

# تحميل قالب البيانات
TEMPLATE_PATH = "assets/The data frame file to be analyzed (1).xlsx"
with open(TEMPLATE_PATH, "rb") as f:
    st.download_button(
        label="📥 تحميل قالب البيانات (Excel)",
        data=f,
        file_name="قالب_البيانات.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# رفع ملف المستخدم
uploaded_file = st.file_uploader("📤 ارفع ملف البيانات", type=["csv", "xlsx"])

if uploaded_file:
    st.success("✅ تم رفع الملف بنجاح.")

    # قراءة الملف حسب النوع
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=';')
    else:
        df = pd.read_excel(uploaded_file)

    st.write("📄 **معاينة البيانات:**")
    st.dataframe(df.head())

    model = load_model()

    if st.button("🚀 بدء التحليل"):
        with st.spinner("🔄 جاري التحليل..."):
            results = analyze_anomalies(df, model)

        st.success("✅ تم الانتهاء من التحليل.")
        st.subheader("📌 النتائج:")
        st.dataframe(results)

        # تصدير الحالات غير السليمة
        download_df = results[results['التصنيف'] != 'سليم']
        excel_file = "output/نتائج_التحليل.xlsx"
        download_df.to_excel(excel_file, index=False)

        with open(excel_file, "rb") as f:
            st.download_button("📥 تحميل النتائج (Excel)", f, file_name="نتائج_التحليل.xlsx")

# تذييل
st.markdown("---")
st.markdown("👨‍💻 **المطور:** مشهور العباس | ☎️ **التواصل:** 00966553339838 | 📅 **آخر تحديث:** 2025-08-07")
