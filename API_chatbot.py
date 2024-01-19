from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import Levenshtein
from datetime import datetime, timedelta
import re
import calendar
import json
import random
import os

app = Flask(__name__)

nltk_data_path = "/opt/render/nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)

model = load_model('chatbot_model2.h5')
words = pickle.load(open('words2.pkl', 'rb'))
classes = pickle.load(open('classes2.pkl', 'rb'))

with open('intents2.json', 'r') as file:
    intention = json.load(file)

df_responses = pd.read_csv('reponses.csv')
lemmatizer = WordNetLemmatizer()

month_mapping = {
    'janvier': '1',
    'février': '2',
    'mars': '3',
    'avril': '4',
    'mai': '5',
    'juin': '6',
    'juillet': '7',
    'août': '8',
    'septembre': '9',
    'octobre': '10',
    'novembre': '11',
    'décembre': '12'
}

english_to_french_month = {
    'January': 'Janvier',
    'February': 'Février',
    'March': 'Mars',
    'April': 'Avril',
    'May': 'Mai',
    'June': 'Juin',
    'July': 'Juillet',
    'August': 'Août',
    'September': 'Septembre',
    'October': 'Octobre',
    'November': 'Novembre',
    'December': 'Décembre'
}

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [1 if word in tokens else 0 for word in vocab]
    return np.array(bow)

def find_similar_word(word, vocab, threshold):
    most_similar_word = min(vocab, key=lambda x: Levenshtein.distance(word, x))
    distance = Levenshtein.distance(word, most_similar_word)
    return most_similar_word if distance <= threshold else None

def pred_class(text, vocab, labels):
    cleaned_text = ' '.join(clean_text(text))
    cleaned_tokens = [find_similar_word(token, vocab, threshold=2) or token for token in cleaned_text.split()]
    cleaned_text = ' '.join(cleaned_tokens)
    date = extract_date(cleaned_text)
    if not cleaned_text.strip():
        return ["erreur"], date
    bow = bag_of_words(cleaned_text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [(idx, res) for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = [labels[r[0]] for r in y_pred]
    return return_list, date

def find_similar_month(month):
    most_similar_month = min(month_mapping.keys(), key=lambda x: Levenshtein.distance(month, x))
    return most_similar_month

def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, '%d/%m/%Y')
        return True
    except ValueError:
        return False

def extract_date(text):
    date_regex = r'(\d{1,2}/\d{1,2}/\d{4})|(\d{1,2} [A-Za-z]+ \d{4})'
    match = re.search(date_regex, text)
    if match:
        date = match.group(0)
        try:
            datetime.strptime(date, '%d/%m/%Y')
            transformed_date = date
        except ValueError:
            try:
                datetime.strptime(date, '%d %B %Y')
                transformed_date = datetime.strptime(date, '%d %B %Y').strftime('%d/%m/%Y')
            except ValueError:
                day, month, year = date.split()
                day = day.lstrip('0')
                month = month.lower()
                if month == 'mois dernier':
                    d = datetime.now() - timedelta(days=30)
                    transformed_date = d.strftime("%d/%m/%Y")
                elif month in ['semaine dernière', 'semaine derniere']:
                    d = datetime.now() - timedelta(days=7)
                    transformed_date = d.strftime("%d/%m/%Y")
                else:
                    most_similar_month = find_similar_month(month) 
                    month = month_mapping.get(most_similar_month, month)
                    transformed_date = f"{day}/{month}/{year}"
                return transformed_date
        return transformed_date
    return None 

def format_date(date):
    day, month, year = date.split('/')
    day = day.lstrip('0')
    month = month.lstrip('0')
    formatted_date = f"{day}/{month}/{year}"
    return formatted_date

def get_response(intents_list, intents_json, date, transformed_date):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            response = random.choice(i["responses"])
            if i["tag"] == "recharge":
                if date:
                    response += " " + date
                break
            elif i["tag"] == "base active":
                if transformed_date:
                    if transformed_date in df_responses['date'].values:
                        day, month, year = transformed_date.split('/')
                        month = english_to_french_month[calendar.month_name[int(month)]]
                        response = f'La base active de {day} {month} {year} est: {df_responses.loc[df_responses["date"] == transformed_date, "BA"].values[0]}'
                    else:
                        response = f"Je n'ai pas trouvé de réponse pour la date: {transformed_date}."
                else:
                    response = "Veuillez préciser la demande avec la date et avec l'année pour obtenir une réponse précise."
                break
            elif i["tag"] == "recharge_amount":
                if transformed_date:
                    if transformed_date in df_responses['date'].values:
                        amount = df_responses.loc[df_responses['date'] == transformed_date, 'montant_recharge'].values[0]
                        day, month, year = transformed_date.split('/')
                        month = english_to_french_month[calendar.month_name[int(month)]]
                        response = f"Le montant de recharge de {day} {month} {year} est: {amount}."
                    else:
                        response = f"Je n'ai pas trouvé de réponse pour la date: {transformed_date}."
                else:
                    response = "Veuillez préciser la demande avec la date et avec l'année pour obtenir une réponse précise."
                break
            elif i["tag"] == "number_of_clients":
                if transformed_date:
                    if transformed_date in df_responses['date'].values:
                        day, month, year = transformed_date.split('/')
                        month = english_to_french_month[calendar.month_name[int(month)]]
                        response = f'Le nombre de client de {day} {month} {year} est: {df_responses.loc[df_responses["date"] == transformed_date, "nombre_client_recharge"].values[0]}'
                    else:
                        response = f"Je n'ai pas trouvé de réponse pour la date: {transformed_date}."
                else:
                    response = "Veuillez préciser la demande avec la date et avec l'année pour obtenir une réponse précise."
                break
    return response

def save_user_message(message):
    with open('user_messages.txt', 'a') as file:
        file.write(message + '\n')

def send_message(msg):
    msg = msg.strip()
    if msg != '':
        transformed_date = extract_date(msg)
        if transformed_date is None:
            if any(keyword in msg.lower() for keyword in ['hier', 'j-1', 'hie', 'ier', 'her']):
                d = datetime.now() - timedelta(days=1)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif any(keyword in msg.lower() for keyword in ['aujourd\'hui', 'aujourd hui', 'aujoud hui', 'aujord hui', 'ajourd hui']):
                transformed_date = format_date(datetime.now().strftime("%d/%m/%Y"))
            elif 'mois dernier' in msg.lower():
                d = datetime.now() - timedelta(days=30)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif 'semaine dernière' in msg.lower() or 'semaine derniere' in msg.lower():
                d = datetime.now() - timedelta(days=7)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
        process_message(msg, transformed_date)

def process_message(msg, transformed_date):
    response = ""
    lowercase_msg = msg.lower()
    intents, date = pred_class(lowercase_msg, words, classes)
    if not intents:
        intents = ["erreur"]
    if date is None:
        date = ""
    if "mois dernier" in lowercase_msg:
        d = datetime.now() - timedelta(days=30)
        transformed_date = format_date(d.strftime("%d/%m/%Y"))
    elif "semaine dernière" in lowercase_msg or "semaine derniere" in lowercase_msg:
        d = datetime.now() - timedelta(days=7)
        transformed_date = format_date(d.strftime("%d/%m/%Y"))

    if intents[0] == "recharge":
        response = get_response(intents, intention, date, transformed_date)
    elif intents[0] == "base active":
        response = get_response(intents, intention, date, transformed_date)
    else:
        response = get_response(intents, intention, date, transformed_date)
    print("You:", msg)
    print("Bot:", response)
    if intents[0] == "erreur":
        save_user_message(msg)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        transformed_date = extract_date(user_message)
        intents, date = pred_class(user_message.lower(), words, classes)
        response = get_response(intents, intention, date, transformed_date)
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)