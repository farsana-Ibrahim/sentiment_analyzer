import nltk
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Speak something...")
    audio = recognizer.listen(source)

try:
    # Convert speech to text
    text = recognizer.recognize_google(audio)

    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)

    # Interpret the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    print(f"Spoken Text: {text}")
    print(f"Sentiment: {sentiment}")
    print("Sentiment Scores:", sentiment_scores)

except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results; {e}")