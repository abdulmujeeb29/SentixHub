from model import preprocess_text, predict, vec 
import joblib 
test_review = [
    'An awful film! It must have been up against some real stinkers to be nominated for the Golden Globe. ',
    'I really like Salman Kahn so I was really disappointed when I seen this movie.',
    "This is the best version (so far) that you will see and the most true to the Bronte work. Dalton is a little tough to imagine as Rochester who Jane Eyre declared ""not handsome"". But his acting overcomes this and Zelah Clark, pretty as she is, is also a complete and believable Jane Eyre. This production is a lengthy watch but well worth it. Nearly direct quotes from the book are in the script and if you want the very first true 'romance' in literature, this is the way to see it. I own every copy of this movie and have read and re-read the original. The filming may seem a little dated now but there will never be another like this."

]

loaded_model = joblib.load('trained_model.pkl')

preprocesed_test_review = [preprocess_text(review) for review in test_review ] 

vectorized_review = vec.transform(preprocesed_test_review)

prediction = loaded_model.predict(vectorized_review)


for i, prediction in enumerate(prediction):
    sentiment = "Negative" if prediction == 0 else "Positive"
    print(f'The sentiment for this review: "{test_review[i]}" is {sentiment}')


