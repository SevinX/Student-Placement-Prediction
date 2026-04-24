import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Klasifikasi Student Placement", layout="wide")


@st.cache_resource 
def load_model():
    model = joblib.load('best_xgb_cls_model.pkl')
    return model

model = load_model()

@st.cache_resource 
def load_reg_model():
    reg_model = joblib.load('best_xgb_reg_model.pkl')
    return reg_model

reg_model = load_reg_model()

st.title("Placement Prediction System")
st.markdown("Masukkan data pada sidebar untuk mendapatkan hasil prediksi")


st.sidebar.header("Form Input Data")
st.sidebar.markdown("Masukkan data:")


with st.sidebar.form("input_form"):

    gender = st.selectbox("Gender", options=['Male', 'Female'])
    branch = st.selectbox("Jurusan", options=['ECE', 'IT', 'CSE', 'CE', 'ME'])
    part_time_job = st.selectbox("Part-time Job?", options=['Yes', 'No'])
    family_income_level = st.selectbox("Family Income Level", options=['Low', 'Medium', 'High'])
    city_tier = st.selectbox("City Tier", options=['Tier 1', 'Tier 2', 'Tier 3'])
    internet_access = st.selectbox("Internet Access?", options=['Yes', 'No'])
    extracurricular_involvement = st.selectbox("Extracurricular Involvement", options=['Low', 'Medium', 'High'])   

    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    twelfth_percentage = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    backlogs = st.number_input("Jumlah Backlogs", min_value=0, value=0, step=1)
    study_hours_per_day = st.number_input("Study Hours per Day", min_value=0.0, value=4.0, step=0.5)
    attendance_percentage = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)

    projects_completed = st.number_input("Projects Completed", min_value=0, value=3, step=1)
    internships_completed = st.number_input("Internships Completed", min_value=0, value=1, step=1)
    coding_skill_rating = st.number_input("Coding Skill Rating (1-5)", min_value=1, max_value=5, value=3, step=1)
    communication_skill_rating = st.number_input("Communication Skill Rating (1-5)", min_value=1, max_value=5, value=3, step=1)
    aptitude_skill_rating = st.number_input("Aptitude Skill Rating (1-5)", min_value=1, max_value=5, value=3, step=1)
    hackathons_participated = st.number_input("Hackathons Participated", min_value=0, value=1, step=1)
    certifications_count = st.number_input("Certifications Count", min_value=0, value=2, step=1)

    sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, value=7.0, step=0.5)
    stress_level = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5, step=1)
    
    submit_button = st.form_submit_button(label="START")


if submit_button:
    st.write("---")
    st.subheader("🔍 Hasil Analisis")
    
    input_data = pd.DataFrame({
        'gender': [gender],
        'branch': [branch],
        'cgpa': [cgpa],
        'tenth_percentage': [tenth_percentage],
        'twelfth_percentage': [twelfth_percentage],
        'backlogs': [backlogs],
        'study_hours_per_day': [study_hours_per_day],
        'attendance_percentage': [attendance_percentage],
        'projects_completed': [projects_completed],
        'internships_completed': [internships_completed],
        'coding_skill_rating': [coding_skill_rating],
        'communication_skill_rating': [communication_skill_rating],
        'aptitude_skill_rating': [aptitude_skill_rating],
        'hackathons_participated': [hackathons_participated],
        'certifications_count': [certifications_count],
        'sleep_hours': [sleep_hours],
        'stress_level': [stress_level],
        'part_time_job': [part_time_job],
        'family_income_level': [family_income_level],
        'city_tier': [city_tier],
        'internet_access': [internet_access],
        'extracurricular_involvement': [extracurricular_involvement]
    })
    
    input_data['cgpa_to_12th_ratio'] = (input_data['cgpa'] * 10) / (input_data['twelfth_percentage'] + 1e-5)
    input_data['total_projects_internships'] = input_data['projects_completed'] + input_data['internships_completed']
    input_data['study_to_sleep_ratio'] = input_data['study_hours_per_day'] / (input_data['sleep_hours'] + 1e-5)

    st.write("**Data yang dimasukkan:**")
    st.dataframe(input_data, use_container_width=True)
    
    try:
        prediction = model.predict(input_data)[0]
        
    
        if prediction == 1 or str(prediction).lower() == 'placed':
            st.success(" **Hasil Prediksi: Selamat anda berkemungkinan besar diterima kerja!**")
            #XGBoost Regression
            try:
                salary_log_pred = reg_model.predict(input_data)[0]
                salary_pred = np.expm1(salary_log_pred)
                st.info(f"**Estimasi Gaji: {salary_pred:.2f} LPA (Lakhs Per Annum)**")
            except Exception as reg_e:
                st.error(f"Terjadi kesalahan saat memprediksi gaji: {reg_e}")
            st.balloons()
        else:
            st.warning(" **Hasil Prediksi: Maaf, data riwayat hidup kamu sepertinya belum cukup untuk dapat diterima kerja :(**")
        
        st.write("---")
        st.write("**Visualisasi Profil Keterampilan Mahasiswa:**")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        

        skill_names = ['Coding Skill', 'Communication Skill', 'Aptitude Skill']
        skill_values = [coding_skill_rating, communication_skill_rating, aptitude_skill_rating]
        

        bars = ax.bar(skill_names, skill_values, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
        ax.set_ylim(0, 5.5) 
        ax.set_ylabel('Rating (1-5)')
        ax.set_title('Perbandingan Skill Mahasiswa')
        
  
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom')
            
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")