from flask import Flask, request, jsonify, send_file
import re
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords
STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    """Endpoint to test if the service is running."""
    return "Test request received successfully. Service is running."

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to handle sentiment prediction requests."""
    # Load models and tools from the Models folder
    sentiment_model = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    vectorizer = pickle.load(open("Models/countVectorizer.pkl", "rb"))

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            uploaded_file = request.files["file"]
            reviews_df = pd.read_csv(uploaded_file)

            predictions_csv, sentiment_graph = handle_bulk_prediction(sentiment_model, scaler, vectorizer, reviews_df, request.form)

            # Send the predictions and sentiment graph as response
            response = send_file(
                predictions_csv,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(sentiment_graph.getbuffer()).decode("ascii")

            return response

        elif "text" in request.json:
            # Single string prediction
            review_text = request.json["text"]
            predicted_sentiment = handle_single_prediction(sentiment_model, scaler, vectorizer, review_text, request.json)

            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def preprocess_text(text, options):
    """Preprocess the text based on the provided options."""
    # Convert to lowercase if option is set
    if options.get("convert_to_lowercase", True):
        text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.split()

    # Remove stopwords if option is set
    if options.get("remove_stopwords", True):
        text = [word for word in text if word not in STOPWORDS]

    # Apply stemming if option is set
    if options.get("use_stemming", True):
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]

    # Apply lemmatization if option is set
    if options.get("use_lemmatization", False):
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]

    # Remove punctuation if option is set
    if options.get("remove_punctuation", True):
        text = [word for word in text if word.isalnum()]

    return " ".join(text)

def handle_single_prediction(model, scaler, vectorizer, text_input, options):
    """Handle single text input prediction."""
    corpus = []
    processed_text = preprocess_text(text_input, options)
    corpus.append(processed_text)
    X_transformed = vectorizer.transform(corpus).toarray()
    X_scaled = scaler.transform(X_transformed)
    prediction_probabilities = model.predict_proba(X_scaled)
    predicted_class = prediction_probabilities.argmax(axis=1)[0]

    return "Positive" if predicted_class == 1 else "Negative"

def handle_bulk_prediction(model, scaler, vectorizer, data, options):
    """Handle bulk prediction from a DataFrame."""
    corpus = []
    for i in range(data.shape[0]):
        review = data.iloc[i]["Sentence"]
        processed_review = preprocess_text(review, options)
        corpus.append(processed_review)

    X_transformed = vectorizer.transform(corpus).toarray()
    X_scaled = scaler.transform(X_transformed)
    prediction_probabilities = model.predict_proba(X_scaled)
    predicted_classes = prediction_probabilities.argmax(axis=1)
    predicted_sentiments = list(map(map_sentiment, predicted_classes))

    data["Predicted sentiment"] = predicted_sentiments
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    sentiment_graph = create_sentiment_distribution_graph(data)

    return predictions_csv, sentiment_graph

def create_sentiment_distribution_graph(data):
    """Create a pie chart showing the distribution of sentiments."""
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    sentiment_counts = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    sentiment_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

def map_sentiment(class_label):
    """Map numeric class labels to sentiment labels."""
    return "Positive" if class_label == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
