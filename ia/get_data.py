import os
import requests
from bs4 import BeautifulSoup

# URL de l'API Wikipédia pour obtenir des articles aléatoires
url = "https://fr.wikipedia.org/wiki/Spécial:Page_au_hasard"

def download_wikipedia_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extraire le texte de l'article
    paragraphs = soup.find_all('p')
    article_text = "\n".join([para.get_text() for para in paragraphs])
    return article_text

def save_to_file(text, filename):
    # Créer le répertoire 'corpus' s'il n'existe pas
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Télécharger et sauvegarder plusieurs articles
for i in range(1000):  # Ajustez le nombre d'articles selon vos besoins
    article_text = download_wikipedia_article(url)
    save_to_file(article_text, f'corpus/article_{i}.txt')

print("Téléchargement terminé.")
