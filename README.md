🎓 Student Performance & Grades Prediction

This project focuses on predicting student academic performance based on demographic, social, and academic attributes. Using a dataset of secondary school students, we apply multiple machine learning models, deep learning, and NLP techniques to explore the factors influencing grades and identify at-risk students.

🔍 Project Overview

Dataset: Portuguese secondary school students (UCI repository).

Target: Final grade (G3) transformed into risk categories (Low, Medium, High).

Objective:

Predict student performance levels.

Identify social, behavioral, and academic features influencing success.

Explore NLP-based student feedback sentiment to assess stress and motivation.

Deploy a trained ML model for real-world usage.

⚙️ Workflow

Data Preprocessing

Column renaming for clarity

Encoding categorical variables (binary + one-hot)

Handling yes/no features as 0/1

Feature scaling with StandardScaler

Feature engineering (social index, alcohol use, free-time ratio)

Risk category extraction from final grades

Models Implemented

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Neural Network (Keras)

NLP Sentiment Analysis

Student feedback analyzed with TextBlob and VADER.

Mapping student sentiment → performance risk.

Evaluation Metrics

Accuracy, Precision, Recall, F1-score

Confusion Matrix & Visualization

Cross-validation for model robustness

Deployment

Best-performing model (Random Forest / SVM) saved with Joblib

Artifacts include: model, scaler, feature columns, numeric columns, and risk labels

Ethical Considerations

No personal identifiers in data

Check for bias in gender and social features

Focus on fairness in predictions

📊 Results

Random Forest & SVM achieved the best performance (~91% accuracy).

Decision Tree underperformed compared to ensemble methods.

VADER outperformed TextBlob in analyzing short, student-like feedback.

🚀 Key Takeaways

Academic success is influenced by a mix of study time, family support, alcohol use, absences, and social behavior.

Sentiment analysis of feedback provides valuable insight into student motivation and stress.

Deployed ML models can serve as early-warning systems to support struggling students.
