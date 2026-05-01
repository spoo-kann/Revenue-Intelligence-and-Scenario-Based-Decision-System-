

# 📊 Revenue Intelligence & Scenario-Based Decision System

An AI-powered data analytics application built using **Streamlit** that helps businesses analyze revenue trends, generate forecasts, and derive actionable insights for smarter decision-making.

---

## 🚀 Features

* 📂 Upload and process business datasets
* 🧹 Data cleaning & validation
* ⚙️ Automated feature engineering
* 🤖 Train multiple ML models
* 🏆 Select best-performing model
* 📈 Revenue forecasting
* 📊 Interactive visualizations (Plotly)
* 🔍 Feature importance analysis
* 📄 Generate downloadable reports (CSV & PDF)
* 🔐 User authentication system

---

## 🛠️ Tech Stack

* **Frontend & App Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualization:** Plotly
* **Forecasting:** Custom ML models
* **PDF Reports:** FPDF
* **Authentication:** Custom Python module

---

## 📁 Project Structure

```
├── app.py                 # Main Streamlit app
├── auth.py                # Authentication system
├── requirements.txt       # Dependencies
├── utils/
│   ├── data_processor.py  # Data cleaning & preprocessing
│   ├── models.py          # Model training & selection
│   ├── forecaster.py      # Forecast generation
│   ├── insights.py        # Feature importance & insights
│   ├── report.py          # Report generation (CSV & PDF)
├── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/spoo-kann/Revenue-Intelligence-and-Scenario-Based-Decision-System-
cd Revenue-Intelligence-and-Scenario-Based-Decision-System-
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📌 Usage

1. Upload your dataset
2. Clean and preprocess data
3. Train models automatically
4. View insights & visualizations
5. Generate forecasts
6. Download reports

---

## 📊 Example Use Cases

* Revenue prediction for businesses
* Sales trend analysis
* Scenario-based planning
* KPI performance tracking
