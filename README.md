# Project Title

StyleSense Fashion Review Recommendation Prediction Pipeline

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Project Overview

This project builds an end-to-end machine learning pipeline that processes multi-modal data including:

Text data: Customer review titles and detailed review text
Numerical data: Customer age and positive feedback counts
Categorical data: Product classifications (clothing ID, division, department, class)

The pipeline uses advanced NLP techniques and ensemble methods to achieve high accuracy in predicting customer recommendations, helping businesses understand customer sentiment and improve product offerings.

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
textblob>=0.17.0
jupyter>=1.0.0
```

### Installation

Step by step explanation of how to get a dev environment running.

List out the steps

1. Clone the repository

git clone https://github.com/TiaZhang92/3rd-nano-dsnd-pipelines-project


2. Create a virtual environment

python -m venv fashion_env
source fashion_env/bin/activate  # On Windows: fashion_env\Scripts\activate

3. Install required packages

pip install -r requirements.txt

4. Download NLTK data (required for text processing)

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

5. Launch Jupyter Notebook

jupyter notebook starter.ipynb

## Testing

Model Validation
The project includes comprehensive testing approaches:
Cross-validation during hyperparameter tuning
Hold-out test set evaluation
Prediction function testing with sample data

Performance Validation
Train/Validation/Test Split: Proper data separation to avoid overfitting
Cross-Validation: 3-fold CV during grid search for robust parameter selection
Multiple Metrics: Accuracy, precision, recall, F1-score, and ROC-AUC for comprehensive evaluation

### Break Down Tests

Usage Example
After training, use the pipeline to predict new recommendations:
result = predict_recommendation(
    title="Great quality dress!",
    review_text="I love this dress. The material is high quality and fits perfectly...",
    age=28,
    pos_feedback_count=0,
    clothing_id=1234,
    division_name="General",
    department_name="Dresses", 
    class_name="Dresses"
)

print(f"Recommendation: {result['recommendation_text']}")
print(f"Confidence: {result['probability']:.2f}")


## Project Instructions

Pipeline Architecture
The machine learning pipeline consists of several integrated components:
1. Data Preprocessing

Numerical: StandardScaler for age and feedback counts
Categorical: OneHotEncoder for product classifications
Text: TF-IDF Vectorization with stop word removal and feature limitation

2. NLP Processing

Text normalization and cleaning
TF-IDF feature extraction from review titles (1,000 features)
TF-IDF feature extraction from review text (3,000 features)
Stop word removal and minimum document frequency filtering

3. Model Training

Random Forest Classifier as the base model
GridSearchCV for hyperparameter optimization
Cross-validation for robust performance estimation

4. Evaluation Metrics

Accuracy, Precision, Recall, F1-Score
ROC-AUC for probability-based evaluation
Confusion Matrix for detailed performance analysis

## Built With
Built With

Scikit-learn - Machine learning library for pipeline creation and model training
Pandas - Data manipulation and analysis
NumPy - Numerical computing foundation
NLTK - Natural language processing toolkit
Matplotlib - Data visualization and plotting
Seaborn - Statistical data visualization
Jupyter Notebook - Interactive development environment
TextBlob - Simplified text processing (for future enhancements)


## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.
