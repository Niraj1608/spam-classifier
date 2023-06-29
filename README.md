# Email Spam Classifier

This Email Spam Classifier is a machine learning model that predicts whether an email is spam or not. It utilizes Multinomial Naive Bayes algorithm, NLTK, Matplotlib, and both CountVectorizer and TfidfVectorizer for feature extraction.

## Description

The Email Spam Classifier is designed to analyze the content of an email and determine whether it is likely to be a spam or legitimate email. It uses a machine learning approach to classify emails based on their textual features.

The project includes the following components:

- **Multinomial Naive Bayes:** The model is trained using Multinomial Naive Bayes, a popular algorithm for text classification, to learn the patterns and characteristics of spam and non-spam emails.

- **NLTK:** The Natural Language Toolkit (NLTK) is a Python library that provides various tools and resources for natural language processing. It is used for preprocessing the email text and extracting relevant features.

- **Matplotlib:** Matplotlib is a plotting library in Python. It is used for visualizing the performance metrics and results of the classifier, such as accuracy, precision, and recall.

- **Feature Extraction with CountVectorizer and TfidfVectorizer:** The project utilizes both CountVectorizer and TfidfVectorizer for feature extraction. CountVectorizer converts the text data into numerical features by counting the occurrences of words, while TfidfVectorizer converts the text data into numerical features based on term frequency-inverse document frequency.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Niraj1608/email-spam-classifier.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your email dataset in CSV format, where each row represents an email and includes a "text" column for the email content and a "label" column for the corresponding label (0 for non-spam, 1 for spam).

2. Train the classifier using CountVectorizer:
   ```
   python train_countvectorizer.py --input dataset.csv --output model_countvectorizer.pkl
   ```

3. Train the classifier using TfidfVectorizer:
   ```
   python train_tfidfvectorizer.py --input dataset.csv --output model_tfidfvectorizer.pkl
   ```

4. After training, you will have two trained models saved as "model_countvectorizer.pkl" and "model_tfidfvectorizer.pkl".

5. Run the web application to classify emails interactively using CountVectorizer:
   ```
   streamlit run app_countvectorizer.py
   ```

   This will launch a local web server. Access the application through your web browser.

6. Run the web application to classify emails interactively using TfidfVectorizer:
   ```
   streamlit run app_tfidfvectorizer.py
   ```

   This will launch a local web server. Access the application through your web browser.

7. Enter the email text in the provided text area and click the "Classify" button to see the prediction.

## Results and Evaluation

The performance of the classifier can be evaluated using various metrics, including accuracy, precision, and recall. These metrics indicate how well the classifier performs in distinguishing between spam and non-spam emails.

The results are visualized using Matplotlib, showing the accuracy, precision, and recall scores. These visualizations can help you understand the performance of the model and make informed decisions.

## License

This project does not have a specific license. All rights reserved.


## Acknowledgements
This project is built upon the contributions and resources from the open-source community, including scikit-learn, NLTK, and Matplotlib.

## contact
If you have any questions or inquiries, please contact nirajprmr1608@example.com.

