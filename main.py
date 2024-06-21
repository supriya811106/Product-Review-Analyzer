import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import base64
import time

# Apply custom CSS
st.markdown("<style>{}</style>".format(open("static/style.css").read()), unsafe_allow_html=True)

# Display logo with resized width
st.image("static/logo.png", use_column_width=True)

# Main header and description
st.markdown(
    "<div class='header'>"
    "<h1>FeedbackFinder - Elevate Your Product with Customer Insights</h1>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='content'>"
    "<p>Welcome to FeedbackFinder, where you can unlock the true potential of customer feedback effortlessly. 😊</p>"
    "<p>FeedbackFinder is a powerful and intuitive tool crafted to help you delve deep into the sentiments and opinions expressed in product reviews. "
    "By leveraging our advanced sentiment analysis models, you can transform raw feedback into actionable insights that drive product innovation and customer satisfaction.</p>"
    "<p class='content-text'>With FeedbackFinder, you can:</p>"
    "<ul>"
    "<li><b>Quickly analyze bulk reviews</b>: Upload a CSV file with multiple reviews and get sentiment predictions at scale.</li>"
    "<li><b>Evaluate individual feedback</b>: Enter a single review to instantly understand the sentiment behind the words.</li>"
    "<li><b>Customize your analysis</b>: Choose from various text preprocessing options to tailor the analysis to your needs.</li>"
    "<li><b>Visualize sentiment distribution</b>: Generate and download graphs to easily interpret the overall customer sentiment.</li>"
    "<li><b>Enhance your product strategy</b>: Use the insights gained to refine your products, address customer concerns, and celebrate what they love.</li>"
    "</ul>"
    "<p>Start exploring the voice of your customers today and make data-driven decisions that elevate your product offerings.</p>"
    "</div>",
    unsafe_allow_html=True
)

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "How It Works", "Feedback"])

with tab1:
    st.markdown("<div class='header'>"
                "<h2>Sentiment Analysis</h2>"
                "</div>",
                unsafe_allow_html=True
    )

    # Model selection
    model_option = st.selectbox(
        "Select a model for sentiment prediction:",
        ("XGBoost", "RandomForest", "LogisticRegression")
    )

    # File uploader and text input for prediction
    uploaded_file = st.file_uploader("Upload a CSV file containing product reviews for bulk prediction:", type="csv")
    user_input = st.text_area("Enter a product review for sentiment prediction:", height=150)

    # Sidebar options for text preprocessing
    st.sidebar.header("Text Preprocessing Options")
    use_stemming = st.sidebar.checkbox("Use Stemming", value=True)
    remove_stopwords = st.sidebar.checkbox("Remove Stopwords", value=True)
    convert_to_lowercase = st.sidebar.checkbox("Convert to Lowercase", value=True)
    remove_punctuation = st.sidebar.checkbox("Remove Punctuation", value=True)
    use_lemmatization = st.sidebar.checkbox("Use Lemmatization", value=False)

    # Prediction endpoint
    prediction_endpoint = "https://feedbackfinder-api.onrender.com/"

    # Prediction button logic
    if st.button("Predict"):
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                file = {"file": uploaded_file}
                params = {
                    "model": model_option,
                    "use_stemming": use_stemming,
                    "remove_stopwords": remove_stopwords,
                    "convert_to_lowercase": convert_to_lowercase,
                    "remove_punctuation": remove_punctuation,
                    "use_lemmatization": use_lemmatization
                }
                response = requests.post(prediction_endpoint, files=file, data=params)
                response_df = pd.read_csv(BytesIO(response.content))

                st.markdown("<h3>Predictions:</h3>", unsafe_allow_html=True)
                st.dataframe(response_df)

                csv = response_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="Predictions.csv" class="predict-button">Download Predictions</a>'
                st.markdown(href, unsafe_allow_html=True)

                if response.headers.get('X-Graph-Exists') == 'true':
                    graph_bytes = base64.b64decode(response.headers.get('X-Graph-Data'))
                    st.image(graph_bytes, caption='Sentiment Distribution', use_column_width=True)

        elif user_input:
            with st.spinner("Processing..."):
                time.sleep(1)
                params = {
                    "text": user_input,
                    "model": model_option,
                    "use_stemming": use_stemming,
                    "remove_stopwords": remove_stopwords,
                    "convert_to_lowercase": convert_to_lowercase,
                    "remove_punctuation": remove_punctuation,
                    "use_lemmatization": use_lemmatization
                }
                response = requests.post(prediction_endpoint, json=params).json()
                st.markdown(f"<h3>Predicted sentiment:</h3> {response['prediction']}", unsafe_allow_html=True)
        else:
            st.warning("Please upload a CSV file or enter some text.")

with tab2:
    st.markdown("<div class='header'>"
                "<h2>How It Works</h2>"
                "</div>",
                unsafe_allow_html=True
    )
    st.markdown("""
        1. **Select a Model**: Choose a machine learning model for sentiment analysis.
        2. **Upload Reviews**: Upload a CSV file with product reviews or enter a single review.
        3. **Preprocess Text**: Select text preprocessing options such as stemming, stopword removal, etc.
        4. **Predict Sentiment**: Get sentiment predictions and download the results.
        5. **Analyze Feedback**: Use the insights to improve your products and services.
    """)

with tab3:
    st.markdown("<div class='header'>"
                "<h2>Feedback</h2></div>",
                unsafe_allow_html=True
    )
    was_helpful = st.selectbox(
        "Did you find our insights useful?",
        ["Please choose an option", "Yes, very helpful", "Somewhat helpful", "Not helpful"],
        key='feedback_select'
    )
    if was_helpful != "Please choose an option":
        feedback = st.text_area("Kindly share any additional comments or suggestions...", key='feedback_text')
        name = st.text_input("Your Name", key='feedback_name')
        email = st.text_input("Your Email", key='feedback_email')
        if st.button("Submit Feedback", key='feedback_button'):
            with st.spinner("Submitting feedback..."):
                time.sleep(2)
                st.success("Thank you for sharing your thoughts!")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Welcome to FeedbackFinder! Our tool leverages advanced machine learning models to analyze sentiments from product reviews. "
    "Upload a CSV file with reviews for bulk predictions, or enter a single review for real-time analysis. "
    "Use our insights to understand customer feedback and enhance your products and services."
)

# Footer
st.markdown("<footer>"
            "<p>FeedbackFinder - Elevate Your Product with Customer Insights</p>"
            "</footer>",
            unsafe_allow_html=True
)
