from django.shortcuts import render
import joblib 
from  model import preprocess_text, predict 
import pandas as pd 
from django.contrib import messages
from itertools import zip_longest  # Import zip_longest from itertools
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


            except Exception as e:
                messages.error(request, "An error occurred while processing the CSV file.")

    return render(request, 'index.html', {
        'reviews': reviews,
        'sentiment': sentiment,
        'zipped_data': zipped_data,
    })
