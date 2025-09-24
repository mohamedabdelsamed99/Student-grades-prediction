# 🎓 Student Performance & Grades Prediction

This project predicts **student academic performance** based on demographic, social, and academic attributes.  
Using a dataset of secondary school students, we apply **machine learning, deep learning, and NLP** techniques to explore the factors influencing grades and identify at-risk students.

---

## 🔍 Project Overview
- **Dataset**: Portuguese secondary school students (UCI repository)  
- **Target**: Final grade (**G3**) → transformed into **risk categories** (Low, Medium, High)  
- **Objectives**:
  - Predict student performance levels  
  - Identify academic, social, and behavioral factors influencing success  
  - Apply **NLP sentiment analysis** on student feedback to assess stress and motivation  
  - Deploy the trained ML model for real-world use  

---

## ⚙️ Workflow

### 📑 Data Preprocessing
- Renamed columns for clarity  
- Encoded categorical variables (binary & one-hot encoding)  
- Converted yes/no features → `0`/`1`  
- Scaled features with **StandardScaler**  
- Feature engineering:
  - Social index  
  - Alcohol consumption  
  - Free-time ratio  
- Extracted **risk categories** from final grades  

### 🤖 Machine Learning Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- Neural Network (**Keras**)
- <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/5a1ea597-4fdf-4c61-91ec-d31c64d6e3c0" />


### 💬 NLP Sentiment Analysis
- Student feedback analyzed with **TextBlob** and **VADER**  
- Sentiment scores mapped to **performance risk levels**
- <img width="545" height="393" alt="image" src="https://github.com/user-attachments/assets/b7c6f160-19bc-4dca-8b56-f3c70a0ce30b" />


### 📊 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix & visualizations  
- Cross-validation for model robustness  

### 🚀 Deployment
- Best-performing model (**Random Forest / SVM**) saved with **Joblib**  
- Artifacts include:
  - Trained model  
  - Scaler  
  - Feature columns  
  - Numeric columns  
  - Risk labels  

### ⚖️ Ethical Considerations
- No personal identifiers included in data  
- Checked for **bias in gender & social features**  
- Focused on fairness and transparency in predictions  

---

## 📊 Results
- **Random Forest & SVM** → Best performance (~91% accuracy)  
- **Decision Tree** → Lower performance compared to ensemble methods  
- **VADER** → Outperformed TextBlob for short, student-like feedback  

---

## 🚀 Key Takeaways
- Academic success is influenced by:
  - Study time  
  - Family support  
  - Alcohol use  
  - Absences  
  - Social behavior  
- Sentiment analysis provides insights into **student motivation & stress**  
- Deployed ML models can act as **early-warning systems** to support struggling students  

---
