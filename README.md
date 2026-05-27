# Weather-Based Outfit Recommendation Chatbot

This project is an AI/ML-based weather outfit recommendation chatbot built using:

- Python
- Flask
- Dialogflow
- Telegram Bot
- OpenWeather API
- Random Forest Machine Learning Model
## Integration
This project is integrated with Google Dialogflow for handling user queries.  
Dialogflow is used for intent detection, and a webhook connects it with the backend logic
## Architecture
User → Dialogflow → Webhook (Python API) → Weather Logic → Response
## Features
- Weather-based outfit suggestions
- Occasion-based recommendations
- Accessories suggestions
- Telegram chatbot integration
- Dialogflow NLP integration

## Tech Stack
- Flask
- Scikit-learn
- Pandas
- Random Forest Classifier
- Dialogflow
- Telegram Bot API

## How to Run

```bash
python model.py
python app.py
```

Then run ngrok:

```bash
ngrok http 5000
```

## Example Inputs
- party outfit in Mumbai
- office outfit in Delhi
- casual outfit in Pune
