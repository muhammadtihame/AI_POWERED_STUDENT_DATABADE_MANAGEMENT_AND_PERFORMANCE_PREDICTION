# 🎓 AI-Powered Student Database Management & Performance Prediction

An intelligent, full-stack web application designed to manage student academic records and predict student performance using machine learning. This system features a user-friendly admin and student portal, automated predictions for pass/fail status and final scores, and seamless backend integration with ML models.

---

## 🚀 Features

- 🔐 Secure student & admin login
- 🧠 AI-powered prediction of:
  - Pass/Fail status
  - Final score
- 📊 Dashboard to manage student records
- 📁 Upload and analyze new student data
- 🧪 Model trained using LightGBM on a large synthetic dataset (5M+ rows)
- ⚙️ RESTful API powered by FastAPI and integrated with ML model

---

## 📁 Project Structure



---

## 🧠 Model Details

- **Algorithm**: LightGBM (Gradient Boosting)
- **Target Variables**: `pass_fail`, `final_score`
- **Features Used**: Age, Attendance, Exam Scores, Project Completion, etc.
- **Training Data**: Synthetic dataset of 5 million rows generated using Faker and pandas

---

## ⚙️ Setup Instructions

### 1. Clone the Repository and Run

```bash
git clone https://github.com/muhammadtihame/AI_POWERED_STUDENT_DATABASE_MANAGEMENT_AND_PERFORMANCE_PREDICTION.git
cd AI_POWERED_STUDENT_DATABASE_MANAGEMENT_AND_PERFORMANCE_PREDICTION
streamlit run app.py

