# Module, Pakete & KI-Bibliotheken
# Module: Es ist mühsam alles selbst zu programmieren
# Was ist ein Modul?
# Eine Datei mit Funktionen und Variablen, die du importieren kannst.
import math
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

print((math.sqrt(25)))  # -> 5.0

# Was ist ein Paket
# Eine Sammlung von Modulen-oft von der Community entwickelt(Z.B. pandas, numpy)

"""
daten = {
    "Name": ["Bernard", "Maya", "John", "Albert", "Isaac"],
    "Alter": [23, 24, 38, 54, 35],
    "Beruf": ["Student", "Sportlerin", "Schauspieler", "Lehrer", "Arzt"],
    "Wohnort": ["Clausthal", "München", "London", "Leipzig", "New York"]
}
df = pd.DataFrame(daten)

# print(df)

# Schreibe eine Funktion, die alle Personen über 23 Jahre herausfiltert.


def person_filtern(df):
    gefiltert = df[["Name", "Alter"]][df["Alter"] > 23]
    return gefiltert


# Wende die Funktion an und gib die gefilterten Personen aus.
gefilterte_personen = person_filtern(df)
print(gefilterte_personen)

# DataFrame nach Alter sortieren
df_nach_alter = df.sort_values(by="Alter")
print(df_nach_alter)

# Dataframe nach Name Sortieren
df_nach_name = df.sort_values(by="Name")
print(df_nach_name)

# Für jeden Eintrag im DataFrame die Quadratwurzel des Alters berechnen


def berechne_quadratwurzel_alter(df):
    df["Alter"] = df["Alter"].apply(lambda x: math.sqrt(x))
    return df


new_df = berechne_quadratwurzel_alter(df)
print(new_df)

"""

# KI-Bibliotheken
# KI-Bibliotheken: Vortrainierte Modelle für Text, Bilder, Sprache
# transformers	Verarbeitung natürlicher Sprache	import transformers
# torch	Grundlage für viele KI-Modelle	import torch

# NumPy	Mathematische Berechnungen, Arrays	import numpy as np
# pandas	Tabellen (Datenrahmen)	import pandas as pd
# matplotlib	Daten visualisieren (Diagramme)	import matplotlib.pyplot as plt
# scikit-learn	Maschinelles Lernen	from sklearn import datasets
# TextBlob	Textanalyse, Sentimentanalyse	from textblob import TextBlob
# TextBlob für die Textanalyse auf Englisch trainiert

# from textblob import TextBlob

# satz = TextBlob("Python is great and I love AI!")
# print(satz.sentiment)

# Lösung transformers
# Das Modell "oliverguhr/german-sentiment-bert" ist speziell für deutsche Texte trainiert.
# Das Modell "tblard/tf-allocine" für Texte auf Französich
# Standard englisches Modell


# Lade ein vortrainiertes Modell für deutsche Sentiment-Analyse
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="oliverguhr/german-sentiment-bert"
)


# Beispieltext auf Deutsch
# text = "Python ist großartig und ich liebe KI!"
texte = [
    "Python ist großartig und ich liebe KI!",
    "Das Wetter ist heute schön.",
    "Ich mag Programmieren.",
    "Künstliche Intelligenz ist die Zukunft."
]


resultate = [sentiment_analyzer(text)[0] for text in texte]

# Dataframe erstellen
df = pd.DataFrame({
    "Text": texte,
    "Sentiment": [result["label"] for result in resultate],
    "Score": [result["score"] for result in resultate]
})
"""
print(df)
def df_nach_label(df):
    df_nach_label_positiv = df[df["Sentiment"] == "positive"]
    df_nach_label_neutral = df[df["Sentiment"] == "neutral"]
    df_nach_label_negativ = df[df["Sentiment"] == "negative"]
    return df_nach_label_positiv, df_nach_label_neutral, df_nach_label_negativ

positiv, neutral, negativ = df_nach_label(df)

label_list = [positiv, neutral, negativ]
for label_df in label_list:
    print(label_df)
"""
# Sortierreihenfolge festlegen
sentiment_order = {"positive": 0, "neutral": 1, "negative": 2}
df["Sentiment_Sort"] = df["Sentiment"].map(sentiment_order)
df_sorted = df.sort_values(by="Sentiment_Sort")
print(df_sorted)

sentiment_analyser_english = pipeline("sentiment-analysis")

texte_english = [
    "Python is great and I love AI!",
    "The weather is nice today.",
    "I enjoy programming.",
    "Artificial intelligence is the future."
]

resultate_english = [sentiment_analyser_english(
    text)[0] for text in texte_english]

df_english = pd.DataFrame({
    "Text": texte_english,
    "Sentiment": [result["label"] for result in resultate_english],
    "Score": [result["score"] for result in resultate_english]
})

sentiment_order_english = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
df_english["Sentiment_Sort"] = df_english["Sentiment"].map(
    sentiment_order_english)
df_english_sorted = df_english.sort_values(by="Sentiment_Sort")
print(df_english_sorted)
# Für französische Texte
sentiment_analysser_french = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)


texte_french = [
    "Python est génial et j'adore l'IA !",
    "Le temps est agréable aujourd'hui.",
    "J'aime programmer.",
    "L'intelligence artificielle est l'avenir."
]

resultate_french = [sentiment_analysser_french(
    text)[0] for text in texte_french]

df_french = pd.DataFrame({
    "Text": texte_french,
    "Sentiment": [result["label"] for result in resultate_french],
    "Score": [result["score"] for result in resultate_french]
})
# 5 stars sentiment très positif
# 4 stars sentiment positif
# 3 stars sentiment neutre
# 2 stars sentiment négatif, 1 star sentiment très négatif

# Extrahiere die Zahl aus "Sentiment" (z.B. "5 stars" -> 5)
df_french["Stars"] = df_french["Sentiment"].str.extract(r"(\d)").astype(int)

# Sortiere nach "Stars" absteigend (höchste zuerst)
df_french_sorted_stars = df_french.sort_values(by="Stars", ascending=False)
print(df_french_sorted_stars)

# Visualisierung der Sentiment-Analyse

# Für Deutsch
df["Sentiment_clean"] = df["Sentiment"].replace({
    "positive": "Positiv", "neutral": "Neutral", "negative": "Negativ"
})
sentiment_counts_de = df["Sentiment_clean"].value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts_de.plot(kind="bar", color=["green", "gray", "red"])
plt.title("Sentiment-Verteilung (Deutsch)")
plt.xlabel("Sentiment")
plt.ylabel("Anzahl Sätze")
plt.show()

# Für Englisch
sentiment_counts_en = df_english["Sentiment"].replace({
    "POSITIVE": "Positiv", "NEUTRAL": "Neutral", "NEGATIVE": "Negativ"
}).value_counts()

plt.figure(figsize=(6, 4))
sentiment_counts_en.plot(kind="bar", color=["green", "gray", "red"])
plt.title("Sentiment-Verteilung (Englisch)")
plt.xlabel("Sentiment")
plt.ylabel("Anzahl Sätze")
plt.show()

# Für Französisch (nach Sternen)
plt.figure(figsize=(6, 4))
df_french["Stars"].value_counts().sort_index(
    ascending=False).plot(kind="bar", color="blue")
plt.title("Sentiment-Verteilung (Französisch, Sterne)")
plt.xlabel("Sterne")
plt.ylabel("Anzahl Sätze")
plt.show()
