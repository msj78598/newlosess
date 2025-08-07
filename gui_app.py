import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import os
from io import BytesIO

# تحميل نموذج Isolation Forest
@st.cache_resource
def load_model():
    return joblib.load("model/isolation_forest.pkl")

# تحليل البيانات
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

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    lof_preds = lof.fit_predict(X_scaled)

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

# واجهة المستخدم
st.set_page_config(page_title="نظام تحليل فاقد الطاقة", layout="wide")
st.title("⚡ نظام اكتشاف حالات الفاقد باستخدام الذكاء الاصطناعي")
st.markdown("📈 **يرجى رفع ملف الأحمال (CSV أو Excel) لتحليله وتحديد الحالات الشاذة**")

uploaded_file = st.file_uploader("📤 ارفع ملف البيانات", type=["csv", "xlsx"])

if uploaded_file:
    st.success("✅ تم رفع الملف بنجاح.")

    # قراءة البيانات
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=';')
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 معاينة البيانات")
    st.dataframe(df.head())

    model = load_model()

    if st.button("🚀 بدء التحليل"):
        with st.spinner("⏳ جاري التحليل باستخدام النموذجين..."):
            results = analyze_anomalies(df, model)

        st.success("🎉 تم الانتهاء من التحليل.")
        st.subheader("📌 النتائج الكاملة:")
        st.dataframe(results)

        # إنشاء ملفات Excel مؤقتة
        def convert_df_to_excel(df_data):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_data.to_excel(writer, index=False, sheet_name='نتائج التحليل')
            output.seek(0)
            return output

        download_df = results[results['التصنيف'] != 'سليم']

        st.markdown("### 📥 تحميل النتائج:")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📥 تحميل كل النتائج (Excel)",
                convert_df_to_excel(results),
                file_name="all_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            st.download_button(
                "📥 تحميل الحالات الشاذة فقط (Excel)",
                convert_df_to_excel(download_df),
                file_name="anomalies_only.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# معلومات المطور
st.markdown("---")
st.markdown("""
👨‍💻 **المطور:** مشهور العباس  
📞 **رقم التواصل:** 00966553339838  
📅 **آخر تحديث:** 07-08-2025
""")
