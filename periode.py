import pandas as pd
import numpy as np
import re
import calendar
import pickle
import json
import random
import tkinter as tk
from tkinter import *
import Levenshtein
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from datetime import datetime, timedelta

nltk.download("punkt")
nltk.download("wordnet")

# Initialisation du lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()

# Chargement du modèle pré-entraîné
model = load_model('chatbot_model2.h5')

# Chargement du vocabulaire et des classes
words = pickle.load(open('words2.pkl', 'rb'))
classes = pickle.load(open('classes2.pkl', 'rb'))

# Charger les données JSON à partir d'un fichier
with open('intents2.json', 'r') as file:
    data = json.load(file)

# Charger les données de réponses à partir d'un fichier CSV
df_responses = pd.read_csv('reponses.csv')

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
    'December': 'Décembre'
}

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

        # Tentative de parsing de la date dans les deux formats
        try:
            datetime.strptime(date, '%d/%m/%Y')
            transformed_date = date
        except ValueError:
            try:
                datetime.strptime(date, '%d %B %Y')
                transformed_date = datetime.strptime(date, '%d %B %Y').strftime('%d/%m/%Y')
            except ValueError:
                # Gérer les cas spéciaux comme 'mois dernier' et 'semaine dernière'
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
                    most_similar_month = find_similar_month(month)  # Obtenir le mois le plus similaire
                    month = month_mapping.get(most_similar_month, month)
                    transformed_date = f"{day}/{month}/{year}"

                return transformed_date

        return transformed_date

    return None  # Retourner None si aucune date valide n'est trouvée

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

    date = extract_date(cleaned_text)  # Extraire la date du texte

    # Vérifier si cleaned_text est vide ou ne contient que des espaces
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

    return return_list, date  # Retourner la liste des intentions détectées et la date extraite

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
                        # Convertir le mois numérique en nom de mois en français
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
                        # Convertir le mois numérique en nom de mois en français
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
                        # Convertir le mois numérique en nom de mois en français
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

# Fonction pour afficher le message de bienvenue
def show_greeting():
    ChatBox.config(state=tk.NORMAL)
    ChatBox.insert(tk.END, " ------ Bienvenue au service Orange -------\n")
    ChatBox.config(state=tk.DISABLED)

# Add a global variable to track pending intent and user input
pending_intent = ""
pending_user_input = ""

# Function to process pending user input and intent when a date is provided
def process_pending_input_and_intent(date):
    global pending_user_input, pending_intent
    if pending_user_input and pending_intent:
        # Process the pending input and intent
        process_message(pending_user_input, date)
        # Clear pending input and intent after processing
        pending_user_input, pending_intent = "", ""

# Function to handle user input
def handle_user_input_and_special_cases(msg):
    global pending_user_input, pending_intent
    transformed_date = extract_date(msg)

    # Check if the user entered an intent without a date
    if any(tag in msg.lower() for tag in ["recharge", "refill", "clients", "ba"]):
        if not transformed_date:
            if 'j-1' in msg.lower() or 'hier' in msg.lower() or 'hie' in msg.lower() or 'ier' in msg.lower() or 'her' in msg.lower():
                # If the user entered a date-related keyword, infer the date as the previous day
                d = datetime.now() - timedelta(days=1)
                transformed_date = format_date(d.strftime("%d/%m/%Y"))
            elif 'aujourd\'hui' in msg.lower() or 'aujourd hui' in msg.lower() or 'aujoud hui' in msg.lower() or 'aujord hui' in msg.lower() or 'ajourd hui' in msg.lower():
               transformed_date = format_date(datetime.now().strftime("%d/%m/%Y"))     
            
              
            else:
                # Save the user input and intent as pending
                pending_user_input = msg
                pending_intent = [tag for tag in ["recharge", "refill", "clients", "ba"] if tag in msg.lower()][0]
                ChatBox.config(state=tk.NORMAL)
                ChatBox.insert(tk.END, "You: " + msg + "\n")  # Display the original message
                ChatBox.insert(tk.END, f"Bot: Please provide a date for the previous query ({pending_intent}).\n")
                ChatBox.config(state=tk.DISABLED)
        else:
            process_pending_input_and_intent(transformed_date)

    # Proceed to process the message
    process_message(msg, transformed_date)

# Fonction pour envoyer un message
def send_message():
    msg = EntryBox.get("1.0", tk.END).strip()
    EntryBox.delete("1.0", tk.END)
    if msg != '':
        if pending_intent:
            process_pending_input_and_intent(msg)
        elif handle_user_input_and_special_cases:
            handle_user_input_and_special_cases(msg)
        else:
            transformed_date = extract_date(msg)
        
        

        process_message(msg, transformed_date)

# Fonction pour traiter le message
def process_message(msg, transformed_date):
    response = ""
    lowercase_msg = msg.lower()  # Convertir le message en minuscules pour le traitement

    # Prédire l'intention et extraire la date
    intents, date = pred_class(lowercase_msg, words, classes)

    # Vérifier si la liste des intentions est vide
    if not intents:
        intents = ["erreur"]

    # Si la date est None, la définir sur une chaîne vide
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
        d= datetime.now()-timedelta(days=2) 
        transformed_date=format_date(d.strftime("%d/%m/%Y"))  
       

    if intents[0] == "recharge":
        response = get_response(intents, data, date, transformed_date)  # Obtenir la réponse en utilisant la date extraite
    elif intents[0] == "base active":
        response = get_response(intents, data, date, transformed_date)
    elif intents[0] == "clients":
        response = get_response(intents, data, date, transformed_date)
    elif intents[0] == "recharge" or intents[0] == "refill" or intents[0] == "clients" or intents[0] == "ba"  :
        # If the user provided a date with a previously pending input, process both
        process_pending_input_and_intent(transformed_date)
        response = get_response(intents, data, date, transformed_date)
    else:
        response = get_response(intents, data, date, transformed_date)    

    ChatBox.config(state=tk.NORMAL)
    ChatBox.insert(tk.END, "You: " + msg + "\n")  # Afficher le message original
    ChatBox.insert(tk.END, "Bot: " + response + "\n")
    ChatBox.config(state=tk.DISABLED)

    if intents[0] == "erreur":
        save_user_message(msg)

# Fonction pour envoyer un message lorsqu'on appuie sur la touche "Entrée"
def send(event=None):
    send_message()

# Créer la fenêtre principale
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Créer la fenêtre de chat
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="lato")
ChatBox.config(state=DISABLED)

# Associer la barre de défilement à la fenêtre de chat
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Créer le bouton pour envoyer le message
SendButton = Button(root, font=("lato", 12, 'bold italic'), text="Envoyer", width="12", height=5,
                    bd=0, bg="#ff7f00", activebackground="#0080ff", fg='#000000',
                    command=send_message, anchor="center", justify="center")

# Créer la zone pour entrer le message
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="lato")
EntryBox.bind("<Return>", send)

# Placer tous les composants à l'écran
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

# Afficher le message de bienvenue
show_greeting()

# Lancer la boucle principale de l'interface graphique
root.mainloop()