Quick Start Guide
Step 1: Install Requirements
bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn joblib

Step 2: Setup Your Data
Create a data folder in your project directory
Place your CSV file with forest fire data in the data folder
Make sure your data has these columns:
X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain, month, day, area

Step 3: Run the System
bash
python main.py
Important: Handling Plots
When you run the code, graphs will pop up automatically. You MUST close each graph window by clicking the 'X' button to let the code continue running. Don't worry - all graphs are also saved automatically in the reports folder.
What Gets Created
The code will automatically create these folders:
models/ - saved machine learning models
reports/ - graphs and results
data/ - your input data goes here

What the Code Does
Data Processing
Cleans your data automatically
Finds and fixes outliers
Converts text months (jan, feb) to numbers
Creates new smart features from existing data
Machine Learning
Tests 4 different models:
Random Forest
SVM
Neural Network
XGBoost
Picks the best one automatically
Shows performance results
Results You'll See
Data cleaning reports - before/after graphs
Model comparisons - which model works best
Feature importance - what factors matter most
Prediction examples - sample fire risk predictions
Business summary - easy-to-understand results

Expected Results
After running, you should see:
85%+ accuracy in predicting fire risk
80%+ fire detection rate
<20% false alarms
Clear explanations of what drives fire risk

Troubleshooting
If you get errors, check that your CSV file is in the data folder
Make sure all required packages are installed
The code will create log files with detailed error information
Graphs must be closed manually for the code to continue

Need Help?
Check the automatically generated log files for detailed error messages. The system is designed to handle most common data issues automatically.