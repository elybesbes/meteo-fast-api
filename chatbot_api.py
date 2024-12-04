import re
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

intent_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["forecast_weather", "current_weather", "other"]

common_cities = [
    "Paris", "Tunis", "Moscow"
]

def get_intent(user_input):
    """Classify the intent of the user input"""
    result = intent_pipeline(user_input, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]

def extract_entities(user_input):
    """Extract city and date from the user input"""
    user_input_lower = user_input.lower()
    city = None
    for city_name in common_cities:
        if city_name.lower() in user_input_lower:
            city = city_name
            break
    
    date_match = re.search(r"(today|tomorrow|\d{4}-\d{2}-\d{2})", user_input)
    date = date_match.group(0) if date_match else None
    
    return city, date

def get_weather(city, date=None):
    api_key = "28ea0fc6ea2501934699dba0be7781bc"
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    
    response = requests.get(url)
    data = response.json()
    
    if "current" in data:
        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        
        if date:
            return f"The weather forecast for {city} on {date} is {temperature}°C with {description}."
        else:
            return f"The current temperature in {city} is {temperature}°C with {description}."
    else:
        return "Sorry, I couldn't retrieve the weather data."

class UserInput(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
async def get_chat_response(user_input: UserInput):
    """Chatbot endpoint to receive and process user input"""

    intent, confidence = get_intent(user_input.query)

    if intent in ["forecast_weather", "current_weather"]:
        city, date = extract_entities(user_input.query)
        
        if city:
            return {"response": get_weather(city, date)}
        else:
            return {"response": "Sorry, I couldn't understand the city. Could you please specify it?"}
    else:
        return {"response": "I'm not sure how to help with that. Could you ask about the weather?"}

# TEST1 : What is the forecast for Moscow 2024-12-15?
# TEST2 : What is the weather in Tunis tomorrow?
# To Run : uvicorn chatbot_api:app --reload