# FeedbackFinder - Product Review Analyzer

![FeedbackFinde](https://github.com/user-attachments/assets/64e7c63b-5a69-49f7-b93d-aacea8ac010b)

FeedbackFinder is a robust tool designed to help you uncover sentiments and opinions within product reviews. Leverage this data to enhance your products and services based on customer feedback.
[Live Demo](https://feedbackfinder-ui.onrender.com)  

## Features

- **Sentiment Analysis**: Predict sentiments (positive or negative) of product reviews using different machine learning models.
- **Bulk Predictions**: Upload a CSV file containing multiple reviews for bulk prediction.
- **Single Review Analysis**: Enter a single product review for instant sentiment prediction.
- **Text Preprocessing**: Customize preprocessing options such as stemming, stopword removal, lowercase conversion, punctuation removal, and lemmatization.
- **Download Predictions**: Export sentiment prediction results as a CSV file.
- **Feedback Collection**: Gather user feedback to improve the application.

## Installation

1. Clone the repository:
    ```bash
    git clone github.com:supriya811106/Product-Review-Analyzer.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Product-Review-Analyzer
    ```
3. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and navigate to `http://localhost:8501` to access the application.

   Or check out the live version here: [FeedbackFinder Live](https://feedbackfinder-ui.onrender.com)

## Usage

### Sentiment Analysis

1. **Select a Model**: Choose a machine learning model for sentiment prediction (XGBoost, RandomForest, LogisticRegression).
2. **Upload Reviews**: Upload a CSV file containing product reviews for bulk prediction or enter a single review in the text area.
3. **Preprocess Text**: Choose text preprocessing options from the sidebar.
4. **Predict Sentiment**: Click the "Predict" button to get sentiment predictions.
5. **Download Results**: Download the prediction results as a CSV file if a file was uploaded.

### How It Works

1. **Select a Model**: Choose a machine learning model for sentiment analysis.
2. **Upload Reviews**: Upload a CSV file with product reviews or enter a single review.
3. **Preprocess Text**: Select text preprocessing options such as stemming, stopword removal, etc.
4. **Predict Sentiment**: Get sentiment predictions and download the results.
5. **Analyze Feedback**: Use the insights to improve your products and services.

### Feedback

1. **Provide Feedback**: Select how helpful you found the insights.
2. **Share Comments**: Optionally, provide additional comments or suggestions.
3. **Submit**: Enter your name and email (optional) and submit your feedback.

## Customization

- **Custom CSS**: Modify the `static/style.css` file to change the look and feel of the application.
- **Logo**: Replace `static/logo.png` with your own logo.

## Acknowledgements

Thank you for using FeedbackFinder! We hope it helps you gain valuable insights from your product reviews.
