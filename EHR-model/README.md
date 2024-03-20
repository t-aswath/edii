<div align=center>
  <img src="https://github.com/bitspaceorg/.github/assets/119417646/577c8581-499e-4cbb-a2f8-e78c643204bc" width="150" alt="Logo"/>
   <h1>EDII-HACKATHON</h1>
  <img src="https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white">
<img src="https://img.shields.io/badge/:bitspace-%23121011?style=for-the-badge&logoColor=%23ffffff&color=%23000000">
<img src="https://img.shields.io/badge/edii-%23121011?style=for-the-badge&color=black">
<img src="https://img.shields.io/badge/iiitdm-%23121011?style=for-the-badge&logoColor=%23ffffff&color=%23000000">
<img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&color=black">
</div>

# Models

For forecasting the risk of major diseases in our [Electronic Health Record (EHR)](https://github.com/t-aswath/EHR) application, we have utilized the following five models:

1. **NVD (Novel Variation Detection)**: This model detects variations in patient data to predict the likelihood of lung cancer.

2. **SK (Sugar Kinetics)**: The SK model predicts the likelihood of diabetes based on glucose metabolism indicators.

3. **MP (Mind Pulse)**: The MP model predicts the likelihood of stroke based on cardiovascular health indicators.

4. **KCD (Kidney Condition Detection)**: This model predicts the likelihood of chronic kidney disease (CKD) based on various health indicators.

5. **OSP (Outlier-Sensitive Predictor)**: The OSP model predicts the presence of heart disease, with sensitivity to outliers in the data.

## Techstack

The tech stack used for developing and deploying these models includes:

- **Python**: Programming language used for model development.
- **TensorFlow**: Deep learning library used for building and training neural network models.
- **Scikit-learn**: Library for machine learning algorithms and tools.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical computing.
- **Matplotlib**: Data visualization library.
- **Seaborn**: Statistical data visualization library based on Matplotlib.
- **Google Colab**: Cloud-based platform used for data analysis, model development, and collaboration.
- **GitHub**: Version control and collaboration platform for code hosting and sharing.

These tools and libraries provide a robust foundation for developing, training, evaluating, and deploying machine learning models for healthcare applications like EHR.

# Novel Variation Detection (NVD) Model-1

## Introduction

The Novel Variation Detection (NVD) model is designed to detect the likelihood of lung cancer based on various demographic and health-related factors. It utilizes a Gaussian Naive Bayes classifier to make predictions.

## Dataset

The model is trained on a dataset containing information about individuals, including gender, age, smoking habits, peer pressure exposure, allergies, wheezing, alcohol consumption, and chest pain. The target variable is whether the individual has lung cancer or not.

## Preprocessing

- Irrelevant features (`YELLOW_FINGERS`, `ANXIETY`, `CHRONIC DISEASE`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `FATIGUE`) were removed from the dataset.
- Gender and lung cancer status were converted to binary variables (0 for female, 1 for male; 0 for no lung cancer, 1 for lung cancer).
- The `COUGHING` feature was removed as part of feature selection.

## Model Training

- The dataset was split into training and testing sets using an 80-20 ratio.
- A Gaussian Naive Bayes classifier was trained on the training set.

## Model Evaluation

- The trained model achieved an accuracy of `85.48%` on the test set.
- Classification report metrics (precision, recall, F1-score, support) were generated for further evaluation.

## Model Serialization

- The trained model was serialized using the pickle library and saved as 'nvd.pkl' for future use.

## Usage

- To use the trained model for prediction, load it using the pickle library and apply it to new data.

## Files

- `nvd.pkl`: Serialized NVD model.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Seaborn
- Google Colab (for dataset loading)

# Kidney Condition Detection (KCD) Model-2

## Introduction

The Kidney Condition Detection (KCD) model is developed to predict the likelihood of chronic kidney disease (CKD) based on various health indicators. It employs a Random Forest classifier for making predictions.

## Dataset

The model is trained on a dataset containing information about patients' age, blood pressure, blood glucose level, blood urea level, sodium level, potassium level, hemoglobin level, white blood cell count, red blood cell count, hypertension (htn) status, appetite (appet) condition, anemia (ane) status, and the classification of CKD.

## Preprocessing

- Irrelevant features (`id`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `pcv`, `ba`, `sc`, `dm`, `cad`, `pe`) were removed from the dataset.
- Binary features (`htn`, `appet`, `ane`, `classification`) were encoded as integers (0 or 1).
- Missing values were filled with zeros.
- Rows with missing values were dropped.

## Model Training

- The dataset was split into training and testing sets using an 80-20 ratio.
- A Random Forest classifier with 1000 estimators was trained on the training set.

## Model Evaluation

- The trained model achieved an accuracy of `90%` on the test set.
- Classification report metrics (precision, recall, F1-score, support) were generated for further evaluation.

## Model Serialization

- The trained model was serialized using the pickle library and saved as 'kcd.pkl' for future use.

## Usage

- To utilize the trained model for prediction, load it using the pickle library and apply it to new data.

## Files

- `kcd.pkl`: Serialized KCD model.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Seaborn
- Google Colab (for dataset loading)

# Mind Pulse (MP) Model-3

## Introduction

The Mind Pulse (MP) model is designed to predict the likelihood of stroke based on various health indicators. It utilizes a Logistic Regression classifier for making predictions.

## Dataset

The model is trained on a dataset containing information about individuals' `gender`, `age`, `hypertension` status, `heart disease` status, `average glucose level`, `BMI` (body mass index), `smoking status`, and `stroke` occurrence.

## Preprocessing

- Irrelevant features (`id`, `ever_married`, `work_type`, `Residence_type`) were removed from the dataset.
- Binary features (`gender`, `smoking_status`) were encoded as integers (0 or 1).
- Missing values were filled with zeros.

## Model Training

- The dataset was split into training and testing sets using an 80-20 ratio.
- A Logistic Regression classifier was trained on the training set.

## Model Evaluation

- The trained model achieved an accuracy of `85%` on the test set.
- Classification report metrics (precision, recall, F1-score, support) were generated for further evaluation.

## Model Serialization

- The trained model was serialized using the pickle library and saved as 'mp.pkl' for future use.

## Usage

- To utilize the trained model for prediction, load it using the pickle library and apply it to new data.

## Files

- `mp.pkl`: Serialized MP model.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Seaborn
- Google Colab (for dataset loading)

# Outlier-Sensitive Predictor (OSP) Model-4

## Introduction

The Outlier-Sensitive Predictor (OSP) model aims to predict the presence of heart disease based on various cardiovascular health indicators. It employs a Random Forest classifier for making predictions.

## Dataset

The model is trained on a dataset containing information about individuals' `age`, `sex`, `chest pain` type (cp), `serum cholesterol level` (chol), `number of major vessels colored by fluoroscopy` (ca), and `thalassemia` type (thal). The target variable is the presence or absence of heart disease.

## Preprocessing

- Irrelevant features (`trestbps`, `fbs`, `restecg`, `thalach`, `exang`, `slope`, `oldpeak`) were removed from the dataset.
- The dataset does not contain missing values, so no imputation was necessary.

## Model Training

- The dataset was split into training and testing sets using an 80-20 ratio.
- A Random Forest classifier with 1000 estimators was trained on the training set.

## Model Evaluation

- The trained model achieved an accuracy of `80%` on the test set.
- Classification report metrics (precision, recall, F1-score, support) were generated for further evaluation.

## Model Serialization

- The trained model was serialized using the pickle library and saved as 'osp.pkl' for future use.

## Usage

- To utilize the trained model for prediction, load it using the pickle library and apply it to new data.

## Files

- `osp.pkl`: Serialized OSP model.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Seaborn
- Google Colab (for dataset loading)

# Sugar Kinetics (SK) Model-5

## Introduction

The Sugar Kinetics (SK) model is designed to predict the likelihood of diabetes based on various glucose metabolism indicators. It employs a Random Forest classifier for making predictions.

## Dataset

The model is trained on a dataset containing information about individuals' `glucose level`, `blood pressure`, `insulin level`, `body mass index` (BMI), and `age`. The target variable is the presence or absence of diabetes.

## Preprocessing

- Irrelevant features (`Pregnancies`, `DiabetesPedigreeFunction`, `SkinThickness`) were removed from the dataset.

## Model Training

- The dataset was split into training and testing sets using an 80-20 ratio.
- A Random Forest classifier with 1000 estimators was trained on the training set.

## Model Evaluation

- The trained model achieved an accuracy of `81%` on the test set.
- Classification report metrics (precision, recall, F1-score, support) were generated for further evaluation.

## Model Serialization

- The trained model was serialized using the pickle library and saved as 'sk.pkl' for future use.

## Usage

- To utilize the trained model for prediction, load it using the pickle library and apply it to new data.

## Files

- `sk.pkl`: Serialized SK model.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Seaborn
- Google Colab (for dataset loading)
