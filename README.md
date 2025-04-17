# Mobile Photo Analyzer

A Streamlit web app for analyzing iPhone photos (resolution, brightness, metadata) and applying machine learning models (KNN, Logistic Regression, Decision Tree) to a mobile dataset with interactive visualizations.

## Features
- Photo Analysis: Upload photos to view resolution, color mode, and brightness, with a width vs. height scatter plot.
- Dataset Analysis: Train ML models on a mobile dataset, with correlation heatmaps and confusion matrices.

## Setup
1. Clone the repository: `git clone https://github.com/your-username/mobile-photo-analyzer.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Place a `mobile_dataset.csv` file in the project root.
4. Run the app: `streamlit run iphoneapp.py`

## Dataset
- File: `mobile_dataset.csv`
- Description: A dataset with mobile phone features (e.g., price, specs).
- Source: Provide your own dataset with compatible columns.

## Technologies
- Python
- Streamlit
- PIL (Pillow)
- Pandas, Matplotlib, Seaborn
- Scikit-learn