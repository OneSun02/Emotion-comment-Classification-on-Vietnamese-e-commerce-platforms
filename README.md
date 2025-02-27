# Emotion Comment Classification on Vietnamese E-Commerce Platforms

## Overview
This project aims to classify customer reviews on Vietnamese e-commerce platforms into sentiment categories. By leveraging Natural Language Processing (NLP) techniques and Machine Learning models, the system predicts whether a comment expresses positive, neutral, or negative sentiments.

## Features
- **Automated Data Collection**: Scrapes customer reviews from e-commerce websites such as Shopee, Tiki, and Lazada.
- **Data Preprocessing**: Cleans and structures raw text data for analysis.
- **Sentiment Labeling**: Automatically assigns sentiment labels to comments (e.g., "Very Dissatisfied" to "Extremely Satisfied").
- **Model Training**: Uses machine learning models to predict the sentiment of new comments.
- **Performance Evaluation**: Measures model accuracy and effectiveness.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, Scikit-learn, TextBlob, BeautifulSoup, NLTK, Matplotlib, Seaborn
- **Machine Learning Models**: Logistic Regression, Random Forest, Support Vector Machine (SVM)
- **NLP Techniques**: TF-IDF, Word Embedding, Sentiment Analysis

## Installation
To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/OneSun02/Emotion-comment-Classification-on-Vietnamese-e-commerce-platforms.git
   cd Emotion-comment-Classification-on-Vietnamese-e-commerce-platforms
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Collection**: Run the web scraper to collect reviews from e-commerce sites.
2. **Data Preprocessing**: Clean and preprocess the text data.
3. **Model Training**: Train machine learning models using labeled data.
4. **Prediction**: Use the trained model to classify new comments.

## Results
- Achieved **89.07% accuracy** in sentiment classification.
- Improved prediction performance with optimized feature engineering and model selection.

## Repository Structure
```
├── data/                   # Contains raw and preprocessed data
├── notebooks/              # Jupyter notebooks for analysis and model training
├── models/                 # Trained machine learning models
├── src/                    # Source code for data processing and model training
├── README.md               # Project documentation
├── requirements.txt        # Required Python dependencies
└── main.py                 # Script to run the classification pipeline
```

## Future Improvements
- Enhance model performance with deep learning (e.g., LSTMs, Transformers).
- Expand dataset with more diverse comments.
- Deploy the model as an API for real-time sentiment analysis.

## Author
- **Phạm Xuân Nhất**  
- GitHub: [OneSun02](https://github.com/OneSun02)

---

Feel free to update the README based on new findings or improvements in the project!
