# Egypt Tech Salary Predictor

A Big Data / Machine Learning project that predicts expected monthly salaries for tech jobs in Egypt using real-world job market data.

## Project Overview

This project started with scraping job postings from Wuzzuf to understand the Egyptian tech job market, then evolved into a cleaner and more practical salary prediction system based on:

- standardized job positions
- years of experience
- historical salary data from multiple prepared datasets

The final system allows the user to choose a job position and years of experience from a Streamlit dashboard, then returns an estimated monthly salary in EGP.

---

## Main Objectives

- Collect real tech job data from the Egyptian market
- Clean and normalize salary/job data
- Standardize job titles into a limited number of categories
- Train a machine learning model for salary prediction
- Deploy the final model using Streamlit

---

## Final Prediction Inputs

The final model uses only:

- `job_title`
- `experience_years`

Target:

- `salary_mid`

This simplified design was chosen intentionally to make the model easier to train, easier to explain, and easier to use in the dashboard.

---

## Final Job Categories

The project currently standardizes jobs into the following categories:

- back end engineer
- front end engineer
- full stack engineer
- data/ai engineer
- software testing engineer
- mobile engineer
- devops engineer
- embedded engineer
- technical support engineer
- cybersecurity engineer
- ui/ux designer

---

## Project Structure

```bash
egypt-tech-salary-predictor/
│
├── data_sources/
│   ├── raw/
│   ├── processed/
│   ├── training/
│   └── merged/
│
├── scripts/
│   ├── create_master_schema.py
│   ├── merge_all_training_files.py
│   ├── normalize_kaggle_generic.py
│   ├── normalize_wuzzuf_training.py
│   ├── prepare_final_model_data.py
│   ├── standardize_job_titles.py
│   └── train_final_model.py
│
├── salary_prediction_model/
│   ├── final_salary_model.pkl
│   └── final_model_metrics.json
│
├── wuzzuf_scraping_pipeline/
│   ├── scrape.py
│   ├── data_cleaning.py
│   └── model_training.py
│
├── dashboard.py
├── README.md
├── requirements.txt
└── .gitignore
D## Data Pipeline

### 1. Wuzzuf Scraping Pipeline

Inside `wuzzuf_scraping_pipeline/`:

- **scrape.py**  
  Scrapes job postings from Wuzzuf using Selenium.

- **data_cleaning.py**  
  Cleans salary and experience fields and prepares training-ready data.

- **model_training.py**  
  Trains the original Wuzzuf-only salary prediction model.

This part was kept in the repository because it represents the original scraping and preprocessing work required for the project.

### 2. Final Data Engineering Pipeline

Inside `scripts/`:

- schema creation
- source normalization
- job title standardization
- merging training datasets
- preparing final model data
- training the final model

---

## Final Model

The final model predicts salary based on:

- **job title**
- **years of experience**

After cleaning the dataset and removing salary outliers, multiple models were tested.

### Best Model

- **Random Forest Regressor**

### Final Evaluation

- **R² = 0.3082**
- **MAE = 12,342 EGP**
- **RMSE = 17,349 EGP**

These results are considered reasonable for a salary prediction task using only two input features.

---

## Dashboard

The Streamlit dashboard allows the user to:

- choose one of the predefined tech positions
- choose years of experience from 0 to 10
- receive an estimated monthly salary in EGP

To improve consistency in the UI, the displayed salary is smoothed so that increasing experience does not result in a lower shown salary.

---

## How to Run the Project

### 1. Clone the repository

git clone https://github.com/youssefehab241/egypt-tech-salary-predictor.git
cd egypt-tech-salary-predictor

2. Create a virtual environment
python -m venv venv

3. Activate the virtual environment
Windows PowerShell
.\venv\Scripts\Activate.ps1
Windows CMD
venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

5. Run the dashboard
python -m streamlit run dashboard.py


### Notes
The final dashboard uses the final trained model stored in:
salary_prediction_model/final_salary_model.pkl
The old root model file was removed to avoid confusion.
The Wuzzuf scraping pipeline is preserved for academic review and grading.
This project does not use a scraping API for Wuzzuf; it uses Selenium-based browser automation.
Future Improvements

Possible next steps for improving the project:

adding more useful features such as location or seniority level
improving job title grouping
testing stronger models such as XGBoost or CatBoost
improving overall model accuracy
enhancing dashboard design and visualizations
