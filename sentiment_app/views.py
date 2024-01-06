from django.shortcuts import render
import joblib 
from  model import preprocess_text, predict 
import pandas as pd 
from django.contrib import messages
from itertools import zip_longest  # Import zip_longest from itertools
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# Create your views here.


def index(request):
    prediction = None
    sentiment = None 
    inputText = None 

    if request.method == 'POST':
        inputText =request.POST['inputText']

        if inputText:

            model =joblib.load('trained_model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')

            preprocessed_inputText = preprocess_text(inputText)
            vectorized_input = vectorizer.transform([preprocessed_inputText])

            prediction = model.predict(vectorized_input)
            sentiment = "Negative" if prediction == 0 else "Positive"

            
        
            # for i,j in enumerate(prediction):
            #     if j == 0:
            #         sentiment = "Negative"
            #     elif j == 1:
            #         sentiment = "Positive"

    return render (request, 'index.html', {
        "prediction" : prediction ,
        "sentiment" : sentiment ,
        "inputText" : inputText 
    })





def analyze_csv(request):
    inputCSV = None
    sentiment = []
    model = joblib.load('trained_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    reviews = []
    zipped_data=[]
    chart_data = []

    if request.method == 'POST':
        inputCSV = request.FILES.get('inputCSV')
        if inputCSV:
            try:
                csv_file = pd.read_csv(inputCSV)

                if csv_file.empty:
                    messages.error(request, "CSV file is empty")

                reviews = csv_file['review'].tolist()  # Extract reviews as a list

                for review in reviews:
                    preprocessed_review = preprocess_text(review)
                    vectorized_review = vectorizer.transform([preprocessed_review])
                    prediction = model.predict(vectorized_review)
                    sentiment.append("Negative" if prediction == 0 else "Positive")

                zipped_data = list(zip_longest(reviews, sentiment, fillvalue=None))

                # Count the number of positive and negative sentiments
                positive_count = sentiment.count('Positive')
                negative_count = sentiment.count('Negative')

                # Create a bar chart
                labels = ['Positive', 'Negative']
                counts = [positive_count, negative_count]

                plt.bar(labels, counts)
                plt.title('Sentiment Proportions')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')

                # Save the plot to a BytesIO object
                image_stream = BytesIO()
                plt.savefig(image_stream, format='png')
                plt.close()

                # Move the stream position to the beginning to be able to read it
                image_stream.seek(0)

                # Embed the image data in the HTML response
                chart_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')

            except Exception as e:
                messages.error(request, "An error occurred while processing the CSV file.")

    return render(request, 'index.html', {
        'reviews': reviews,
        'sentiment': sentiment,
        'zipped_data': zipped_data,
        'chart_data': chart_data,
    })
