import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import bcrypt
import os
import uuid
from datetime import datetime

# Apply custom CSS for blue background, white login bars, and footer styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E3A8A;  /* Deep blue background */
        color: white;  /* Adjust text color for readability */
    }
    .stTextInput > label, .stSlider > label, .stSelectbox > label, .stDateInput > label {
        color: white;  /* Make labels white */
    }
    .stTextInput > div > input, .stSelectbox > div > select {
        color: black;  /* Ensure input text is readable */
        background-color: #FFFFFF;  /* White background for inputs including login bars */
    }
    .stButton > button {
        background-color: #4CAF50;  /* Green buttons for contrast */
        color: white;
    }
    .stExpander {
        background-color: #2B4A9B;  /* Slightly lighter blue for expanders */
    }
    .stDataFrame {
        background-color: #FFFFFF;  /* White background for DataFrame */
        color: black;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #FFD700;  /* Gold color for footer text */
        margin-top: 20px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# File paths
USER_FILE = "users.json"
STUDENT_FILE = "students.csv"
FEATURE_NAMES_FILE = "feature_names.pkl"  # Ensure this file is in the same directory

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'user' not in st.session_state:
    st.session_state.user = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'performance_pred' not in st.session_state:
    st.session_state.performance_pred = None
if 'marks_pred' not in st.session_state:
    st.session_state.marks_pred = None

# Load pre-trained models and feature names
try:
    classifier_model = pickle.load(open('lgb_classifier.pkl', 'rb'))
    regressor_model = pickle.load(open('lgb_regressor.pkl', 'rb'))
    feature_names = pickle.load(open(FEATURE_NAMES_FILE, 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Ensure 'lgb_classifier.pkl', 'lgb_regressor.pkl', and 'feature_names.pkl' are in the same directory.")
    st.stop()

# Parse JSON grades
def parse_course_grades(s):
    try:
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("CourseGrades JSON must be a list.")
        grades = []
        for item in arr:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        grades.append(value)
                    else:
                        raise ValueError(f"Invalid grade value: {value}")
            else:
                raise ValueError(f"Invalid item in CourseGrades: {item}")
        if not grades:
            return np.nan
        return np.mean(grades)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in CourseGrades: {e}")
        return np.nan
    except ValueError as e:
        st.error(f"Error parsing CourseGrades: {e}")
        return np.nan

# Initialize user file
def init_users():
    if not os.path.exists(USER_FILE):
        users = {
            "admin": {
                "password": bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "role": "admin"
            }
        }
        with open(USER_FILE, 'w') as f:
            json.dump(users, f)

# Initialize student file with 'GradeAvg'
def init_students():
    if not os.path.exists(STUDENT_FILE):
        df = pd.DataFrame(columns=[
            'student_id', 'name', 'DateOfBirth', 'EnrollmentDate', 'LastLoginDate', 'CourseGrades',
            'Attendance (%)', 'GradeAvg', 'CreditHours', 'Major', 'Residency', 'FinancialAid', 'PandemicEffect',
            'performance', 'marks'
        ])
        df.to_csv(STUDENT_FILE, index=False)

# Load users
def load_users():
    init_users()
    with open(USER_FILE, 'r') as f:
        return json.load(f)

# Load students with column initialization including 'GradeAvg'
def load_students():
    init_students()
    df = pd.read_csv(STUDENT_FILE)
    expected_cols = ['student_id', 'name', 'DateOfBirth', 'EnrollmentDate', 'LastLoginDate', 'CourseGrades',
                     'Attendance (%)', 'GradeAvg', 'CreditHours', 'Major', 'Residency', 'FinancialAid', 'PandemicEffect',
                     'performance', 'marks']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

# Save users
def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)

# Save students
def save_students(df):
    df.to_csv(STUDENT_FILE, index=False)

# Prepare input data for prediction
def prepare_input_data(student_data):
    # Convert dates
    dob = pd.to_datetime(student_data['DateOfBirth'])
    enroll_date = pd.to_datetime(student_data['EnrollmentDate'])
    last_login = pd.to_datetime(student_data['LastLoginDate'])
    age_at_enroll = (enroll_date - dob).days / 365.0
    days_since_login = (pd.Timestamp("2025-04-09") - last_login).days
    course_grades_avg = parse_course_grades(student_data['CourseGrades'])

    # Create a DataFrame with the same columns as training
    input_df = pd.DataFrame(columns=feature_names)

    # Fill numerical features
    input_df['Attendance (%)'] = [student_data['Attendance (%)']]
    input_df['CourseGradesAvg'] = [course_grades_avg]
    input_df['GradeAvg'] = [student_data['GradeAvg']]
    input_df['CreditHours'] = [student_data['CreditHours']]
    input_df['AgeAtEnroll'] = [age_at_enroll]
    input_df['DaysSinceLastLogin'] = [days_since_login]

    # Fill categorical features with one-hot encoding
    for col in ['Major', 'Residency', 'FinancialAid', 'PandemicEffect']:
        val = student_data[col]
        for category in [c.split('_')[1] for c in feature_names if c.startswith(col + '_')]:
            input_df[f"{col}_{category}"] = [1 if val == category else 0]

    # Fill any remaining columns with 0 (e.g., unseen categories)
    input_df = input_df[feature_names].fillna(0)
    return input_df.values  # Return as a numpy array for prediction

# Predict performance and marks
def predict_performance(data):
    try:
        pred = classifier_model.predict(data)[0]
        prob = classifier_model.predict_proba(data)[0]
        labels = ['Low', 'Average', 'High']
        return labels[pred], prob
    except Exception as e:
        st.error(f"Error in performance prediction: {e}")
        return None, None

def predict_marks(data):
    try:
        return regressor_model.predict(data)[0]
    except Exception as e:
        st.error(f"Error in marks prediction: {e}")
        return None

# Login page
def login_page():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        users = load_users()
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]["password"].encode('utf-8')):
            st.session_state.user = username
            st.session_state.role = users[username]["role"]
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password")

    # Add "Created by" footer
    st.markdown('<div class="footer">CREATED BY MOHAMMAD TIHAME</div>', unsafe_allow_html=True)

# Dashboard page
def dashboard_page():
    st.title("Student Performance Dashboard")
    st.write(f"Welcome, {st.session_state.user} ({st.session_state.role.capitalize()})")
    
    if st.button("Logout"):
        st.session_state.page = "login"
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()

    students_df = load_students()

    if st.session_state.role == "admin":
        # Admin functionalities
        st.subheader("Manage Students")
        with st.expander("Add New Student"):
            with st.form("add_student_form"):
                name = st.text_input("Student Name")
                student_id = str(uuid.uuid4())[:8]
                dob = st.date_input("Date of Birth", value=datetime(2000, 1, 1))
                enroll_date = st.date_input("Enrollment Date", value=datetime(2020, 1, 1))
                last_login = st.date_input("Last Login Date", value=datetime(2025, 4, 1))
                course_grades = st.text_input("Course Grades (JSON e.g., '[{\"C1\": 85}, {\"C2\": 90}]')", value='[{\"C1\": 85}, {\"C2\": 90}]')
                attendance = st.slider("Attendance (%)", 0, 100, 80, step=1)
                grade_avg = st.slider("Grade Average", 0, 100, 75, step=1)
                credit_hours = st.slider("Credit Hours", 0, 24, 12, step=1)
                major = st.selectbox("Major", ['ComputerScience', 'Engineering', 'Arts'])
                residency = st.selectbox("Residency", ['OnCampus', 'OffCampus'])
                financial_aid = st.selectbox("Financial Aid", ['Yes', 'No'])
                pandemic_effect = st.selectbox("Pandemic Effect", ['Affected', 'NotAffected'])
                add_button = st.form_submit_button("Add Student")

                if add_button:
                    student_data = {
                        'DateOfBirth': dob.strftime('%Y-%m-%d'),
                        'EnrollmentDate': enroll_date.strftime('%Y-%m-%d'),
                        'LastLoginDate': last_login.strftime('%Y-%m-%d'),
                        'CourseGrades': course_grades,
                        'Attendance (%)': attendance,
                        'GradeAvg': grade_avg,
                        'CreditHours': credit_hours,
                        'Major': major,
                        'Residency': residency,
                        'FinancialAid': financial_aid,
                        'PandemicEffect': pandemic_effect
                    }
                    input_data = prepare_input_data(student_data)
                    performance, _ = predict_performance(input_data)
                    marks = predict_marks(input_data)

                    new_student = pd.DataFrame([{
                        'student_id': student_id,
                        'name': name,
                        'DateOfBirth': student_data['DateOfBirth'],
                        'EnrollmentDate': student_data['EnrollmentDate'],
                        'LastLoginDate': student_data['LastLoginDate'],
                        'CourseGrades': student_data['CourseGrades'],
                        'Attendance (%)': attendance,
                        'GradeAvg': grade_avg,
                        'CreditHours': credit_hours,
                        'Major': major,
                        'Residency': residency,
                        'FinancialAid': financial_aid,
                        'PandemicEffect': pandemic_effect,
                        'performance': performance,
                        'marks': marks
                    }])
                    students_df = pd.concat([students_df, new_student], ignore_index=True)
                    save_students(students_df)

                    users = load_users()
                    users[student_id] = {
                        "password": bcrypt.hashpw(student_id.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                        "role": "student"
                    }
                    save_users(users)
                    st.info(f"New student added!\nUsername: {student_id}\nPassword: {student_id}")

        with st.expander("Edit Student"):
            student_ids = students_df['student_id'].tolist()
            selected_id = st.selectbox("Select Student ID", student_ids)
            if selected_id:
                student = students_df[students_df['student_id'] == selected_id].iloc[0]
                with st.form("edit_student_form"):
                    name = st.text_input("Student Name", value=student['name'])
                    dob = st.date_input("Date of Birth", value=pd.to_datetime(student['DateOfBirth']) if pd.notna(student['DateOfBirth']) else datetime(2000, 1, 1))
                    enroll_date = st.date_input("Enrollment Date", value=pd.to_datetime(student['EnrollmentDate']) if pd.notna(student['EnrollmentDate']) else datetime(2020, 1, 1))
                    last_login = st.date_input("Last Login Date", value=pd.to_datetime(student['LastLoginDate']) if pd.notna(student['LastLoginDate']) else datetime(2025, 4, 1))
                    course_grades = st.text_input("Course Grades (JSON)", value=student['CourseGrades'] if pd.notna(student['CourseGrades']) else '[{\"C1\": 85}, {\"C2\": 90}]')
                    attendance = st.slider("Attendance (%)", 0, 100, int(student['Attendance (%)']) if pd.notna(student['Attendance (%)']) else 80, step=1)
                    grade_avg = st.slider("Grade Average", 0, 100, int(student['GradeAvg']) if pd.notna(student['GradeAvg']) else 75, step=1)
                    credit_hours = st.slider("Credit Hours", 0, 24, int(student['CreditHours']) if pd.notna(student['CreditHours']) else 12, step=1)
                    major = st.selectbox("Major", ['ComputerScience', 'Engineering', 'Arts'], index=['ComputerScience', 'Engineering', 'Arts'].index(student['Major']) if pd.notna(student['Major']) else 0)
                    residency = st.selectbox("Residency", ['OnCampus', 'OffCampus'], index=['OnCampus', 'OffCampus'].index(student['Residency']) if pd.notna(student['Residency']) else 0)
                    financial_aid = st.selectbox("Financial Aid", ['Yes', 'No'], index=['Yes', 'No'].index(student['FinancialAid']) if pd.notna(student['FinancialAid']) else 0)
                    pandemic_effect = st.selectbox("Pandemic Effect", ['Affected', 'NotAffected'], index=['Affected', 'NotAffected'].index(student['PandemicEffect']) if pd.notna(student['PandemicEffect']) else 0)
                    edit_button = st.form_submit_button("Update Student")

                    if edit_button:
                        student_data = {
                            'DateOfBirth': dob.strftime('%Y-%m-%d'),
                            'EnrollmentDate': enroll_date.strftime('%Y-%m-%d'),
                            'LastLoginDate': last_login.strftime('%Y-%m-%d'),
                            'CourseGrades': course_grades,
                            'Attendance (%)': attendance,
                            'GradeAvg': grade_avg,
                            'CreditHours': credit_hours,
                            'Major': major,
                            'Residency': residency,
                            'FinancialAid': financial_aid,
                            'PandemicEffect': pandemic_effect
                        }
                        input_data = prepare_input_data(student_data)
                        performance, _ = predict_performance(input_data)
                        marks = predict_marks(input_data)

                        students_df.loc[students_df['student_id'] == selected_id, [
                            'name', 'DateOfBirth', 'EnrollmentDate', 'LastLoginDate', 'CourseGrades',
                            'Attendance (%)', 'GradeAvg', 'CreditHours', 'Major', 'Residency', 'FinancialAid',
                            'PandemicEffect', 'performance', 'marks'
                        ]] = [name, student_data['DateOfBirth'], student_data['EnrollmentDate'], student_data['LastLoginDate'],
                              student_data['CourseGrades'], attendance, grade_avg, credit_hours, major, residency, financial_aid,
                              pandemic_effect, performance, marks]
                        save_students(students_df)
                        st.success("Student updated successfully.")

        # View all students
        st.subheader("All Students")
        st.dataframe(students_df)

        # Add "Created by" footer for admin dashboard
        st.markdown('<div class="footer">CREATED BY MOHAMMAD TIHAME</div>', unsafe_allow_html=True)

    else:
        # Student mode
        st.subheader("Your Performance")
        student = students_df[students_df['student_id'] == st.session_state.user]
        if not student.empty:
            student = student.iloc[0]
            st.write(f"Name: {student['name']}")
            st.write(f"Attendance: {student['Attendance (%)']}%")
            st.write(f"Grade Average: {student['GradeAvg']}")
            st.write(f"Credit Hours: {student['CreditHours']}")
            st.write(f"Major: {student['Major']}")
            st.write(f"Residency: {student['Residency']}")
            st.write(f"Financial Aid: {student['FinancialAid']}")
            st.write(f"Pandemic Effect: {student['PandemicEffect']}")

            # Buttons for predictions
            if st.button("Predict Performance Category"):
                student_data = {
                    'DateOfBirth': student['DateOfBirth'],
                    'EnrollmentDate': student['EnrollmentDate'],
                    'LastLoginDate': student['LastLoginDate'],
                    'CourseGrades': student['CourseGrades'],
                    'Attendance (%)': float(student['Attendance (%)']),
                    'GradeAvg': float(student['GradeAvg']),
                    'CreditHours': float(student['CreditHours']),
                    'Major': student['Major'],
                    'Residency': student['Residency'],
                    'FinancialAid': student['FinancialAid'],
                    'PandemicEffect': student['PandemicEffect']
                }
                input_data = prepare_input_data(student_data)
                performance, prob = predict_performance(input_data)
                if performance:
                    st.session_state.performance_pred = (performance, prob)

            if st.button("Predict Future Marks"):
                student_data = {
                    'DateOfBirth': student['DateOfBirth'],
                    'EnrollmentDate': student['EnrollmentDate'],
                    'LastLoginDate': student['LastLoginDate'],
                    'CourseGrades': student['CourseGrades'],
                    'Attendance (%)': float(student['Attendance (%)']),
                    'GradeAvg': float(student['GradeAvg']),
                    'CreditHours': float(student['CreditHours']),
                    'Major': student['Major'],
                    'Residency': student['Residency'],
                    'FinancialAid': student['FinancialAid'],
                    'PandemicEffect': student['PandemicEffect']
                }
                input_data = prepare_input_data(student_data)
                marks = predict_marks(input_data)
                if marks:
                    st.session_state.marks_pred = marks

            # Display predictions if available
            if st.session_state.performance_pred:
                performance, prob = st.session_state.performance_pred
                st.write(f"Performance: **{performance}**")
                fig, ax = plt.subplots()
                ax.bar(['Low', 'Average', 'High'], prob, color=['#FF9999', '#66B2FF', '#99FF99'])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                st.pyplot(fig)

            if st.session_state.marks_pred:
                st.write(f"Predicted Marks: **{st.session_state.marks_pred:.2f}/100**")

        else:
            st.error("No data found for this student.")

        # Add "Created by" footer for student dashboard
        st.markdown('<div class="footer">CREATED BY MOHAMMAD TIHAME</div>', unsafe_allow_html=True)

# Main app logic
if st.session_state.page == "login":
    login_page()
else:
    dashboard_page()