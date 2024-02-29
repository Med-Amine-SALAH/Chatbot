# Importation des bibliothèques
from flask import Flask, request, jsonify
from keras.models import load_model
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

# Chemin vers les données nltk
nltk_data_path = "/opt/render/nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)

# Chargement du modèle et des données nécessaires
model = load_model('chatbot_model2.h5')
words = pickle.load(open('words2.pkl', 'rb'))
classes = pickle.load(open('classes2.pkl', 'rb'))

# Chargement des intentions à partir du fichier JSON
with open('intents2.json', 'r') as file:
    intention = json.load(file)

# Chargement des réponses à partir du fichier CSV
df_responses = pd.read_csv('reponses.csv')

lemmatizer = WordNetLemmatizer()

# Mapping des mois
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
    'décembre': '12',
    'jan': '1',
    'fév': '2',
    'mar': '3',
    'avr': '4',
    'mai': '5',
    'jun': '6',
    'jul': '7',
    'aoû': '8',
    'sep': '9',
    'oct': '10',
    'nov': '11',
    'déc': '12'
}

# Dictionnaire pour mapper les noms de mois anglais aux noms de mois français
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
    'December': 'Décembre',
    'Jan': 'Janvier',
    'Feb': 'Février',
    'Mar': 'Mars',
    'Apr': 'Avril',
    'May': 'Mai',
    'Jun': 'Juin',
    'Jul': 'Juillet',
    'Aug': 'Août',
    'Sep': 'Septembre',
    'Oct': 'Octobre',
    'Nov': 'Novembre',
    'Dec': 'Décembre'
}

# Fonction pour nettoyer le texte
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Fonction pour créer le sac de mots
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

# Fonction pour trouver le mois le plus similaire
def find_similar_month(month):
    most_similar_month = None
    min_distance = float('inf')
    for mapped_month, mapped_num in month_mapping.items():
        distance = Levenshtein.distance(month, mapped_month)
        if distance < min_distance:
            min_distance = distance
            most_similar_month = mapped_month
    return most_similar_month

# Fonction pour vérifier si une date est valide
def is_valid_date(date_str):
    try:
        datetime.strptime(date_str, '%d/%m/%Y')
        return True
    except ValueError:
        return False

# Fonction pour extraire la date du texte
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

# Fonction pour formater la date
def format_date(date):
    day, month, year = date.split('/')
    day = day.lstrip('0')
    month = month.lstrip('0')
    formatted_date = f"{day}/{month}/{year}"
    return formatted_date

# Fonction pour trouver le mot le plus similaire
def find_similar_word(word, vocab, threshold):
    most_similar_word = None
    min_distance = float('inf')
    for vocab_word in vocab:
        distance = Levenshtein.distance(word, vocab_word)
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            most_similar_word = vocab_word
    return most_similar_word

# Fonction pour prédire la classe de l'intention
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
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list, date

# Fonction pour obtenir la réponse en fonction de l'intention
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
                        response = 'La base active de ' + day + ' ' + month + ' ' + year + ' est: ' + str(
                            df_responses.loc[df_responses['date'] == transformed_date, 'BA'].values[0])
                    else:
                        response = f"Je n'ai pas trouvé de réponse pour la date: {transformed_date}."
                else:
                    response = "Veuillez préciser la demande avec la date et avec l'année pour obtenir une réponse précise."
                break
            elif i["tag"] == "recharge_amount":
                if transformed_date:
                    if transformed_date in df_responses['date'].values:
                        amount = str(df_responses.loc[df_responses['date'] == transformed_date, 'montant_recharge'].values[0])
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
                        response = 'Le nombre de client de ' + day + ' ' + month + ' ' + year + ' est: ' + str(
                            df_responses.loc[df_responses['date'] == transformed_date, 'nombre_client_recharge'].values[0])
                    else:
                        response = f"Je n'ai pas trouvé de réponse pour la date: {transformed_date}."
                else:
                    response = "Veuillez préciser la demande avec la date et avec l'année pour obtenir une réponse précise."
                break
    return response

# Fonction pour sauvegarder le message de l'utilisateur
def save_user_message(message):
    with open('user_messages.txt', 'a') as file:
        file.write(message + '\n')

# Ajouter une variable globale pour suivre l'intention en attente et l'entrée de l'utilisateur
pending_intent = ""
pending_user_input = ""

# Fonction pour traiter l'entrée utilisateur en attente et l'intention lorsque une date est fournie
def process_pending_input_and_intent(date):
    global pending_user_input, pending_intent
    if pending_user_input and pending_intent:
        response = process_message(pending_user_input, date)
        pending_user_input, pending_intent = "", ""
        return jsonify({'response': response})

# Fonction pour gérer l'entrée utilisateur
def handle_user_input_and_special_cases(msg):
    global pending_user_input, pending_intent
    transformed_date = extract_date(msg)
    response = ""
    if any(tag in msg.lower() for tag in ["recharge", "montant de recharge", "refill", "clients", "nombre de recharge clients", "nombre de clients", "ba", "base active"]):
        if not transformed_date:
            if 'j-1' in msg.lower() or 'hier' in msg.lower() or 'hie' in msg.lower() or 'ier' in msg.lower() or 'her' in msg.lower():
                d = datetime.now() - timedelta(days=1)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif 'aujourd\'hui' in msg.lower() or 'aujourd hui' in msg.lower() or 'aujoud hui' in msg.lower() or 'aujord hui' in msg.lower() or 'ajourd hui' in msg.lower():
                transformed_date = format_date(datetime.now().strftime("%d/%m/%Y"))
            elif 'j-7' in msg.lower() or 'semaine derniere' in msg.lower() or 'semaine dernière' in msg.lower():
                d = datetime.now() - timedelta(days=7)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif 'j-30' in msg.lower() or 'mois dernier' in msg.lower():
                d = datetime.now() - timedelta(days=30)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif 'j-360' in msg.lower() or 'année dernière' in msg.lower() or 'annee derniere' in msg.lower():
                d = datetime.now() - timedelta(days=365)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            else:
                pending_user_input = msg
                pending_intent = [tag for tag in ["recharge", "montant de recharge", "refill", "clients", "nombre de recharge clients", "nombre de clients", "ba", "base active"] if tag in msg.lower()][0]
                return jsonify({'response': f"Veuillez indiquer une date ({pending_intent})."})
        else:
            response = process_message(msg, transformed_date)
    response = process_message(msg, transformed_date)
    return jsonify({'response': response})

# Fonction pour traiter le message
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
    elif "année dérniére" in lowercase_msg or "annee derniere" in lowercase_msg:
        d = datetime.now() - timedelta(days=365)
        transformed_date = format_date(d.strftime("%d/%m/%Y"))
    elif "avant hier" in lowercase_msg:
        d = datetime.now()-timedelta(days=2) 
        transformed_date=format_date(d.strftime("%d/%m/%Y"))
    if intents[0] == "recharge":
        response = get_response(intents, intention, date, transformed_date)
    elif intents[0] == "base active":
        response = get_response(intents, intention, date, transformed_date)
    elif intents[0] == "clients":
        response = get_response(intents, intention, date, transformed_date)
    elif intents[0] in ["recharge", "refill", "clients", "ba"]:
        process_pending_input_and_intent(transformed_date)
        response = get_response(intents, intention, date, transformed_date)
    else:
        response = get_response(intents, intention, date, transformed_date)   
    if os.getenv("FLASK_ENV") == "development":
        print("Response:", response) 
    if intents[0] == "erreur":
        save_user_message(msg)
    return response

# Route pour l'API du chatbot
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if user_message != '':
            if pending_intent:
                response = process_pending_input_and_intent(user_message)
            elif handle_user_input_and_special_cases(user_message):
                response = handle_user_input_and_special_cases(user_message)
            else:
                transformed_date = extract_date(user_message)
                response = process_message(user_message, transformed_date)
        return response
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

# Exécution de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)