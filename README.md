# Dataset Poisoning Simulation & Defense

This is a student-level Machine Learning project that demonstrates the vulnerabilities of machine learning models to **Data Poisoning Attacks** and how to defend against them using **Anomaly Detection**.

The project is built entirely in Python using `scikit-learn` and provides an easy-to-use, interactive `Streamlit` dashboard to represent the findings.

---

## 🎯 Project Objective
The goal is to show that when malicious (poisoned) data is introduced into a training dataset, the accuracy of a normal classification model drops. By using an unsupervised anomaly detector (Isolation Forest), we can find and remove these suspicious rows to **recover** the model's accuracy.

---

## 🚀 How to Run the Project

1. **Clone the repository** (or download the files to your local machine).
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate the Dataset**:
   Run the dataset script to fetch the Iris dataset and save it locally as `dataset.csv`:
   ```bash
   python make_dataset.py
   ```
4. **Start the Application**:
   Launch the Streamlit web dashboard:
   ```bash
   streamlit run app.py
   ```
5. **Open your browser** and go to `http://localhost:8501`.

---

## 🧠 How It Is Implemented

The project is divided into two simple Python scripts to keep it understandable.

### 1. `make_dataset.py` (Dataset Generation)
This script simply loads the standard Iris dataset from `scikit-learn` and saves it as a local `dataset.csv` file. This simulates having a real local dataset rather than downloading one dynamically in the app.

### 2. `app.py` (The Main Dashboard & ML Logic)
This file handles everything from the User Interface (UI) to training the models. It is divided into three navigation tabs:

#### **Function / Section 1: Data Upload & Exploratory Data Analysis (EDA)**
- **Purpose**: Mimics a real-world data analysis step.
- **What it does**: It reads `dataset.csv` using `pandas.read_csv()`, previews the first 10 rows, and displays the dataset shape, missing values, and statistical summary using `df.describe()`.

#### **Function / Section 2: Model Training (Clean Baseline)**
- **Purpose**: Establishes a baseline accuracy before any attack happens.
- **What it does**: 
  - Splits the clean dataset into training and testing sets using `train_test_split()`.
  - Trains a `RandomForestClassifier` on the clean data.
  - Predicts the test set and calculates the original clean **Accuracy**.

#### **Function / Section 3: Poisoning Analysis (Simulation & Defense)**
This is the core experiment of the project.
- **Step A: Poisoning the Data**:
  - We take a percentage of the training data (e.g., 15%) and corrupt it.
  - **Label Flipping**: We change the target label of some rows to an incorrect class.
  - **Backdoor Attack**: We add a high numerical value (+3.0) to a feature and force its label to a specific class.
  - We then train a *Poisoned Model* on this corrupted data, which usually results in a drop in accuracy.
- **Step B: Defense via Isolation Forest**:
  - We combine the features and labels and pass them to an `IsolationForest` (an Unsupervised Anomaly Detection algorithm).
  - The Isolation Forest flags rows that look mathematically "suspicious" or out of place (which catches our poisoned data).
- **Step C: Recovery**:
  - We remove the suspicious rows flagged by the Isolation Forest.
  - We train a *Recovered Model* on the cleaned dataset and show that the accuracy bounces back up.
- **Visualization**:
  - A `matplotlib` bar chart visually represents the Clean vs. Poisoned vs. Recovered accuracy for easy presentation.

---

## 📊 Why This Project is Easy to Represent
- **Interactive UI**: The Streamlit interface makes it easy to click buttons and see results instantly without running terminal commands during a presentation.
- **Visual Evidence**: The final bar chart clearly proves the hypothesis: *Poisoning lowers accuracy, cleaning restores it.*
- **No Complex Deep Learning**: By sticking to Random Forests and Isolation Forests, the math and logic are easy to explain during a viva or presentation.
