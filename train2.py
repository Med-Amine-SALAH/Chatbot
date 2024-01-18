import pandas as pd
import numpy as np
import json
import string
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
import pickle

nltk.download("punkt")
nltk.download("wordnet")

# Initialisation du lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()

# Charger les données JSON à partir d'un fichier
with open('intents2.json', 'r') as file:
    data = json.load(file)

# Création des listes
words = []
classes = []
doc_X = []
doc_y = []

# Parcourir toutes les intentions
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemmatiser tous les mots du vocabulaire et les convertir en minuscule
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Trier le vocabulaire et les classes par ordre alphabétique et prendre le set pour éliminer les doublons
words = sorted(set(words))
classes = sorted(set(classes))

# Charger les données de réponses à partir d'un autre fichier (par exemple, un fichier Excel)
df_responses = pd.read_excel('reponses.xlsx')

# Assurez-vous que la colonne contenant les réponses est nommée "Reponse"
reponses = df_responses

# Liste pour les données d'entraînement
training = []
out_empty = [0] * len(classes)

# Création du modèle Bag of Words
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Sauvegarde des classes et du vocabulaire
pickle.dump(classes, open('classes2.pkl', 'wb'))
pickle.dump(words, open('words2.pkl', 'wb'))

# Définition des dimensions d'entrée et de sortie du modèle
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

# Création du modèle neural
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

# Entraînement du modèle
hist = model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

# Sauvegarde du modèle
model.save('chatbot_model2.h5')
print("Model is created.")