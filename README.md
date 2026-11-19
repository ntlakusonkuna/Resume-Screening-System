# Resume Screening System

## Project Overview

The Resume Screening System is a machine learning application developed to automate aspects of the initial resume screening process. It analyses resume documents, predicts potential job categories, extracts selected candidate information and compares resume content against a supplied job description.

The application combines a trained machine learning model with an interactive Streamlit interface.

## Project Objectives

- Develop a machine learning model that classifies resumes into occupational categories.
- Extract text from PDF and Word documents.
- Identify selected candidate information, including contact details, education and skills.
- Compare resumes against job descriptions using text similarity.
- Allow users to analyse multiple resumes.
- Generate screening reports in Excel and PDF formats.
- Maintain a local screening history.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorisation
- Random Forest Classifier
- Streamlit
- SQLite
- PyPDF
- Python-docx
- ReportLab
- Openpyxl
- Jupyter Notebook

## Machine Learning Methodology

The project followed a structured machine learning workflow:

1. Data loading and inspection.
2. Data cleaning and duplicate removal.
3. Text preprocessing, including lowercasing, punctuation removal, whitespace normalisation and stopword removal.
4. Feature extraction using TF-IDF vectorisation.
5. Stratified training and testing split.
6. Training and comparison of multiple classification algorithms.
7. Hyperparameter tuning.
8. Model evaluation and error analysis.
9. Saving and integrating the trained model into the Streamlit application.

### Dataset Preparation

The original dataset contained 2,484 resume records across 24 occupational categories.

After removing duplicate resumes and an empty resume record, 2,481 records remained.

The data was divided into:
- 80% training data.
- 20% testing data.

TF-IDF was used to convert resume text into numerical features, with a maximum of 5,000 features and unigram and bigram representation.

### Model Evaluation

Several classification algorithms were evaluated, including:

- Logistic Regression
- Linear Support Vector Machine
- Multinomial Naive Bayes
- Random Forest

The tuned Random Forest model achieved the following results on the held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 77.26% |
| Weighted F1-score | 75.80% |
| Macro F1-score | 73.00% |

The selected model used 500 estimators and a minimum split size of 5.

## Application Features

### 1. Dashboard
Displays screening statistics and recent screening activity.

### 2. Multiple Resume Uploads
Supports uploading multiple PDF and DOCX resumes for analysis.

### 3. Resume Classification
Predicts the three most likely occupational categories and displays their confidence scores.

### 4. Candidate Information Extraction
Extracts selected information, including:
- Candidate name (estimated from the first non-empty line).
- Email address.
- Phone number.
- Education-related text.
- Recognised skills.

### 5. Job Description Matching
Compares resume content with a supplied job description using cosine similarity on TF-IDF features.

### 6. Candidate Comparison
Displays screening results in a comparison table and can rank candidates by job description similarity.

### 7. Screening History
Stores screening records in a local SQLite database.

### 8. Report Generation
Allows users to export screening results and history into Excel and PDF formats.

## Project Structure

```text
Resume-Screening-System/
│
├── app.py
├── README.md
├── .gitignore
├── screening_history.db
│
├── models/
│   └── resume_screening_model.pkl
│
└── [Jupyter Notebook and other project files]
```

The screening history database is excluded from version control through `.gitignore`.

## Installation and Usage

### Requirements

- Python 3.11 or compatible version.
- The trained model file.
- Required Python libraries.

### Install Dependencies

```bash
pip install streamlit pandas scikit-learn nltk joblib pypdf python-docx reportlab openpyxl
```

### Run the Application

Navigate to the project directory and execute:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Limitations and Considerations

- The classification model is restricted to the 24 occupational categories represented in its training dataset.
- Categories that are not represented in the training data may receive an incorrect or approximate prediction.
- Candidate information extraction uses basic rules and regular expressions and may not always identify information correctly.
- Job description matching is based on TF-IDF cosine similarity. It measures textual similarity, not actual candidate competence or suitability.
- The dataset contains imbalanced occupational categories, which can affect classification performance.
- The application is intended to support initial screening and should not independently determine hiring decisions.

## Future Improvements

Potential future development includes:

- More advanced candidate information extraction.
- Improved handling of previously unseen job categories.
- More comprehensive job description matching.
- Improved evaluation of classification performance.
- Additional visual analytics and reporting.
- More robust document extraction.

## Disclaimer

This project is an educational and portfolio project demonstrating machine learning, natural language processing and application development. Predictions and similarity scores should be interpreted as automated indicators and reviewed by a human decision-maker.

## Author

Developed as a Data Science and Machine Learning portfolio project.