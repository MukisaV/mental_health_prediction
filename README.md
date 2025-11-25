# mental_health_prediction
README.md — Mental Health Treatment Prediction (OSMI Dataset)
Predicting Mental Health Treatment-Seeking Behavior in the Tech Industry Using Machine Learning
This project develops and evaluates machine-learning models to predict whether individuals working in the tech industry seek mental-health treatment. It applies rigorous preprocessing, model comparison, fairness assessment, and explainability techniques (SHAP & LIME) to ensure transparency and ethical use.
Dataset
•	Source: OSMI Mental Health in Tech Survey (2014)
•	Rows: 1,251
•	Target: treatment_Yes
•	Features: Demographics, workplace factors, employer support, mental-health history
•	Preprocessing:
o	Cleaning & normalization
o	One-hot encoding
o	Train/Test split
o	Standard scaling

Models Used
Model	Purpose	Notes
Logistic Regression	Baseline	Balanced and stable
RidgeCV (Final Model)	Best performance	Highest generalization
LassoCV	Feature shrinkage	High recall
Naïve Bayes	Probabilistic baseline	Weak accuracy
Decision Tree	Interpretability baseline	Simple rule extraction


Key Results
•	Best Model: RidgeCV
•	Accuracy: ~82.5%
•	ROC-AUC: 0.898
•	Top Predictors:
o	Work interference
o	Family history
o	Care options availability
o	Employer mental-health benefits
o	Leave policy difficulty
Fairness Assessment
Fairness across Gender and Region:
•	Slight precision gap for males -> slightly more false positives
•	Regional performance varies mildly (US > Europe > Others)
•	No extreme disparities or harmful bias
•	Model does not heavily rely on sensitive attributes
Explainability
SHAP
•	Global importance (bar plot)
•	Global distribution (beeswarm)
•	Local explanation (force plot)
LIME
•	Instance-level interpretability
•	Shows which features drive a prediction
•	Validates SHAP and Decision Tree insights

Exported Models
Saved using joblib.dump():
•	ridgecv_model.pkl
•	logistic_regression_model.pkl
•	lassocv_model.pkl
•	naive_bayes_model.pkl
•	decision_tree_model.pkl
Loading a Saved Model
import joblib

model = joblib.load("ridgecv_model.pkl")
preds = model.predict(X_test_scaled)

Installation
Create environment:
pip install -r requirements.txt
requirements.txt
numpy
pandas
scikit-learn
shap
lime
matplotlib
seaborn
joblib

Usage
Run the notebook:
jupyter notebook ML_Semester_Project.ipynb
Or execute the Python script version:
python ml_semester_project.py

Project Structure
/data
    OSMI_cleaned_phase3_ready.csv
    OSMI_encoded_phase4_ready.csv

/notebooks
    Mukisa Vicent ML_Semester_Project.ipynb

/models
    ridgecv_model.pkl
    logistic_regression_model.pkl
    lassocv_model.pkl
    naive_bayes_model.pkl
    decision_tree_model.pkl

/scripts
    ml_semester_project.py

README.md
requirements.txt
LICENSE

Academic Information
Field	Detail
Programme	Master of Science in Computer Science (MCSC)
Course Unit	MCS 7103: Machine Learning
Academic Year	2025/2026
Project Title	Predicting Mental Health Treatment-Seeking Behavior in the Tech Industry Using Machine Learning
Student	MUKISA Vicent

License
This project is released under the MIT License.
You are free to use, modify, and distribute the code with attribution.

