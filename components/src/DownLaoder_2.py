import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse
import subprocess
from collections import deque
import time

# Configuration
OUTPUT_FOLDER = "downloads"
MAX_DURATION = 8  # Durée maximale des vidéos à télécharger (en secondes)
MAX_URLS_TO_VISIT = 1000 # Limite pour éviter de crawler l'intégralité d'internet. Ajuster selon les besoins.
CRAWL_DELAY = 0.5 # Délai (en secondes) entre chaque requête, pour être respectueux.

# Mots-clés recherchés dans le nom des fichiers vidéo ou leur contexte.
KEYWORDS = ["merci", "salut", "bonjour", "vendredi", "lsf", "langue des signes"] # Ajout de 'lsf' et 'langue des signes'

# Fichier pour enregistrer les URLs visitées, pour pouvoir reprendre en cas d'interruption.
VISITED_URLS_FILE = "visited_urls.txt"


def get_video_duration(video_url):
    """Obtient la durée d'une vidéo en utilisant ffprobe."""
    try:
        response = requests.get(video_url, stream=True, timeout=10) # Timeout ajouté
        response.raise_for_status() # Vérifie si la requête a réussi (statut 200)

        with open("temp_video", "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                break

        result = subprocess.run(
            [
                "ffprobe", "-i", "temp_video", "-show_entries",
                "format=duration", "-v", "quiet", "-of", "csv=p=0"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15 # Timeout ajouté ici aussi
        )
        os.remove("temp_video")

        try:
            return float(result.stdout.decode("utf-8").strip())
        except ValueError:
            print(f"Impossible de convertir la durée en float pour {video_url}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur de requête pour {video_url} : {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout ffprobe pour {video_url}")
        return None
    except FileNotFoundError:
        print("ffprobe n'est pas installé ou n'est pas dans le PATH.")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de l'obtention de la durée pour {video_url}: {e}")
        return None


def download_video(url, output_folder=OUTPUT_FOLDER):
    """Télécharge une vidéo."""
    filename = url.split("/")[-1]
    filepath = f"{output_folder}/{filename}"

    # Vérifie si le fichier existe déjà
    if os.path.exists(filepath):
        print(f"Le fichier {filepath} existe déjà.  Skipping...")
        return

    try:
        print(f"Téléchargement de {filename} depuis {url}...")
        response = requests.get(url, stream=True, timeout=20)  # Ajout timeout
        response.raise_for_status()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Vidéo téléchargée : {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur de téléchargement de {filename} depuis {url} : {e}")
    except Exception as e:
        print(f"Erreur inattendue lors du téléchargement de {filename} depuis {url}: {e}")


def is_video_relevant(url):
    """Vérifie si l'URL ou son nom de fichier contient un des mots-clés."""
    filename = url.split("/")[-1].lower()
    return any(keyword in url.lower() or keyword in filename for keyword in KEYWORDS)


def crawl_website(seed_url):
    """Crawle un site web en largeur et télécharge les vidéos pertinentes."""

    visited_urls = load_visited_urls()  # Charger les URLs visitées depuis le fichier
    urls_to_visit = deque([seed_url]) # Utilisation d'une queue pour un parcours en largeur
    num_urls_visited = 0

    while urls_to_visit and num_urls_visited < MAX_URLS_TO_VISIT:
        url = urls_to_visit.popleft()

        if url in visited_urls:
            continue

        print(f"Exploration de : {url}")
        visited_urls.add(url)
        save_visited_urls(visited_urls) # Sauvegarder après chaque visite.
        num_urls_visited += 1

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10) # Ajout d'un timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, "html.parser")

            # Recherche des balises <video>
            for video in soup.find_all("video"):
                src = video.get("src")
                if src:
                    full_url = urljoin(url, src)
                    if is_video_relevant(full_url):
                        print(f"Vidéo potentielle trouvée : {full_url}")
                        duration = get_video_duration(full_url)
                        if duration is not None and duration <= MAX_DURATION:
                            download_video(full_url)
                        else:
                            print(f"Vidéo ignorée (durée : {duration:.2f} secondes ou durée inconnue) : {full_url}")

            # Recherche des liens (<a>)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                 # Recherche des vidéos via les liens
                if href.endswith((".mp4", ".webm", ".avi", ".mov")):  # Ajout d'extensions courantes
                    if is_video_relevant(full_url):
                        print(f"Vidéo potentielle trouvée via un lien : {full_url}")
                        duration = get_video_duration(full_url)
                        if duration is not None and duration <= MAX_DURATION:
                            download_video(full_url)
                        else:
                            print(f"Vidéo ignorée (durée : {duration:.2f} secondes ou durée inconnue) : {full_url}")



                # Ajouter les liens internes à la queue, mais seulement ceux du même domaine.
                parsed_url = urlparse(url)
                parsed_full_url = urlparse(full_url)

                if parsed_url.netloc == parsed_full_url.netloc and full_url not in visited_urls:
                    urls_to_visit.append(full_url)

        except requests.exceptions.RequestException as e:
            print(f"Erreur de requête pour {url} : {e}")
        except Exception as e:
            print(f"Erreur lors de l'exploration de {url} : {e}")

        time.sleep(CRAWL_DELAY)  # Respect du délai

    print(f"Crawling terminé. {num_urls_visited} URLs visitées.")


def load_visited_urls():
    """Charge les URLs visitées depuis le fichier."""
    try:
        with open(VISITED_URLS_FILE, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()


def save_visited_urls(visited_urls):
    """Sauvegarde les URLs visitées dans le fichier."""
    try:
        with open(VISITED_URLS_FILE, "w") as f:
            for url in visited_urls:
                f.write(url + "\n")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des URLs visitées : {e}")


# Point d'entrée
if __name__ == "__main__":
    # URL de départ (un site qui a potentiellement des liens vers des vidéos LSF)
    #seed_url = "https://www.youtube.com/channel/UCwL9-W3Chb_Ej5Nnf215cTQ" # Exemple : Chaine Youtube LSF
    #seed_url = "https://www.elix-lsf.fr/" # autre exemple
    #seed_url = "https://www.google.com/search?q=dictionnaire+vid%C3%A9o+langue+des+signes+fran%C3%A7aise&sca_esv=ae1c882eac2a8cbb&rlz=1C1GCEU_frFR1074FR1074&sxsrf=AHTn8zrYasoF8y7my3ZCERf1aWQPeRM4Ag%3A1742984413076&ei=3dTjZ9izBMnX7M8P24f1qQM&oq=dictionnaire+vid%C3%A9o+langue+des+signes+fra&gs_lp=Egxnd3Mtd2l6LXNlcnAiKWRpY3Rpb25uYWlyZSB2aWTDqW8gbGFuZ3VlIGRlcyBzaWduZXMgZnJhKgIIADIFECEYoAEyBRAhGKABMgUQIRigATIFECEYnwUyBRAhGJ8FSPIOULoGWOgKcAB4AZABAJgBlgGgAZ4EqgEDMi4zuAEByAEA-AEBmAIFoAL0A8ICBBAAGEeYAwDiAwUSATEgQIgGAZAGCJIHAzIuM6AH4CGyBwMxLjO4B-4D&sclient=gws-wiz-serp" # exemple pour tester sur un site connu
    seed_url = "https://dico.lsfb.be/?s=bonjour"
    print("Début du crawl du web...")
    crawl_website(seed_url)
    print("Crawling terminé.")