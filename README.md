#  Wuzzuf Salary Predictor - Big Data & ML Project

This project is a comprehensive data science pipeline that scrapes job data from **Wuzzuf**, cleans it, and trains a **Machine Learning** model to predict salaries in the Egyptian tech market.
# live demo
Check out the interactive dashboard here:
**[Wuzzuf Salary Predictor Live](https://wuzzuf-salary-predictor-uareuhdosjaappmn8hcramy.streamlit.app/)**
# Project Stages
1. **Scraping:** Extracted 1700+ job listings using Selenium.
2. **Preprocessing:** Handled missing values and removed outliers (yearly and fake salaries).
3. **Modeling:** Trained a Random Forest Regressor using TF-IDF for text features.
4. **Dashboard:** Built an interactive UI with Streamlit.

## File Structure
* `scrape.py`: Selenium script for data collection.
* `data_cleaning.py`: Script for filtering and outlier removal.
* `model_training.py`: ML pipeline (Training & Evaluation).
* `dashboard.py`: Streamlit application code.
* `salary_prediction_model.pkl`: The trained AI model.

##  How to run locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run dashboard.py`
