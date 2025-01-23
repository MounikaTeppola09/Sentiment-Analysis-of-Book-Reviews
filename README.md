# Sentiment-Analysis-of-Book-Reviews

## Overview
This project focuses on sentiment analysis of book reviews to understand reader preferences. By leveraging machine learning, the project classifies reviews into three categories: positive, negative, and neutral. The insights gained can assist authors, publishers, and retailers in enhancing their understanding of readers and tailoring strategies for better engagement.

## Features
- Sentiment Classification: Classifies book reviews into positive, negative, or neutral sentiments.
- Advanced Machine Learning Models: Includes Naive Bayes, Logistic Regression, Support Vector Machines (SVM), Long Short-Term Memory (LSTM), and ensemble techniques like Voting and Stacking classifiers.
- Interactive Notebooks: Provides detailed visualizations and analyses of results.
- Preprocessing Techniques: Utilizes text cleaning, tokenization, lemmatization, and feature extraction (TF-IDF).
- Performance Evaluation: Includes metrics like precision, recall, F1-score, confusion matrices, and ROC curves.

## Dataset
- **Source**: The dataset is sourced from Kaggle and includes book reviews and metadata from Goodreads and Amazon.
- **Size**: 4,400 entries with 15 columns, including attributes such as book title, author, genres, ratings, and sentiment labels.
- **Challenges**: Handled imbalanced sentiment distribution and ensured data quality during preprocessing.

## Getting Started

### Installation
Clone the repository:
```bash
git clone https://github.com/MounikaTeppola09/Sentiment-Analysis-of-Book-Reviews.git
cd Sentiment-Analysis-Books-Reviews


Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project
#### Jupyter Notebook:
Open the `Main_Code.ipynb` file in Jupyter Notebook or Google Colab.  
Run the cells step-by-step to preprocess the data, train models, and evaluate performance.

#### Dataset:
The dataset file `Dataset.csv` is included in the repository. Ensure it is in the same directory as the notebook.

## Outputs
- **Visualizations**: Accuracy plots, confusion matrices, and ROC curves for all models.
- **Performance Metrics**: Includes detailed metrics for Naive Bayes, SVM, LSTM, and ensemble models.

## Machine Learning Models

### Baseline Models:
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

### Advanced Models:
- Long Short-Term Memory (LSTM)

### Ensemble Techniques:
- Voting Classifier
- Stacking Classifier

## Tools and Frameworks
- **Python Libraries**:
  - Scikit-learn: For baseline models and evaluation.
  - TensorFlow: For building and training LSTM models.
  - NLTK: For text preprocessing.
- **Platform**: Google Colab for training and visualization.

## Results

### Best Model:
The ensemble model achieved the highest overall performance with balanced predictions and strong AUC scores:
- **Class 0 (Negative)**: AUC = 0.97
- **Class 1 (Neutral)**: AUC = 0.89
- **Class 2 (Positive)**: AUC = 0.88

### LSTM Performance:
LSTM showed good accuracy (~72%) but struggled with overfitting, performing well on training data but less so on unseen data.

### Baseline Models:
- **Logistic Regression**: Outperformed Naive Bayes and SVM in accuracy and recall, with an overall weighted F1-score of 0.71.
- **Naive Bayes**: Achieved ~70.7% accuracy, performing well for the negative class but less so for neutral and positive classes.
- **SVM**: Struggled to differentiate neutral and positive classes, indicating a bias toward neutral predictions.

## Lessons Learned
- **Preprocessing Importance**: Text cleaning and feature extraction significantly impact results.
- **Model Selection**: Ensemble models provide a balanced approach, leveraging strengths of multiple classifiers.
- **Challenges**: Addressed overfitting in LSTM and computational limitations during training.

## Future Work
- Incorporate metadata for enhanced predictions.
- Expand the dataset to include more diverse reviews.
- Explore transformer-based models like BERT for improved sentiment understanding.

## Requirements
The project requires the following Python libraries:
- scikit-learn
- tensorflow
- nltk
- pandas
- matplotlib

Install them using:
```bash
pip install -r requirements.txt
```
```

This is ready for direct use in your GitHub repository’s README!
