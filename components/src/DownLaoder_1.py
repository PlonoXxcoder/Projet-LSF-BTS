import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urljoin, urlparse
import subprocess

# Fonction pour vérifier la durée de la vidéo
def get_video_duration(video_url):
    try:
        # Télécharge temporairement les métadonnées de la vidéo
        response = requests.get(video_url, stream=True, timeout=10)
        with open("temp_video", "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                break  # Télécharge uniquement un petit morceau de la vidéo

        # Utilisation de ffprobe pour obtenir les informations de la vidéo
        result = subprocess.run(
            [
                "ffprobe", "-i", "temp_video", "-show_entries",
                "format=duration", "-v", "quiet", "-of", "csv=p=0"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15
        )
        os.remove("temp_video")  # Supprimer le fichier temporaire
        try:
            return float(result.stdout.decode("utf-8").strip())
        except ValueError:
            print("Impossible de convertir la durée en float.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'obtention de la durée de la vidéo : {e}")
        return None
    except subprocess.TimeoutExpired:
        print("Timeout ffprobe.")
        return None
    except FileNotFoundError:
        print("ffprobe n'est pas installé ou n'est pas dans le PATH.")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de l'obtention de la durée de la vidéo : {e}")
        return None


# Fonction pour télécharger une vidéo
def download_video(url, output_folder="downloads"):
    filename = url.split("/")[-1]
    filepath = f"{output_folder}/{filename}"
    if os.path.exists(filepath):
        print(f"Le fichier {filepath} existe déjà.  Skipping...")
        return
    try:
        print(f"Téléchargement de {filename}...")
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Vidéo téléchargée : {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement de {filename} : {e}")
    except Exception as e:
        print(f"Erreur inattendue lors du téléchargement de {filename} : {e}")


# Fonction pour crawler un site web
def crawl_website(base_url):
    visited_urls = set()
    video_urls = set()

    def crawl(url):
        if url in visited_urls:
            return
        print(f"Exploration de : {url}")
        visited_urls.add(url)
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Trouver les vidéos via balises <video>
            for video in soup.find_all("video"):
                src = video.get("src")
                if src:
                    full_url = urljoin(base_url, src)
                    video_urls.add(full_url)

            # Trouver toutes les vidéos se terminant par les extensions vidéo courantes
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(base_url, href)

                if full_url.lower().endswith((".mp4", ".webm", ".avi", ".mov")):
                    video_urls.add(full_url)


                # Explorer les liens internes du site
                if base_url in full_url:  # Vérifie si c'est une URL interne
                    crawl(full_url)
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de l'exploration de {url} : {e}")
        except Exception as e:
            print(f"Erreur lors de l'exploration de {url} : {e}")

    crawl(base_url)
    return video_urls

# URL de base du site
base_url = "https://dico.lsfb.be/?s=bonjour"

# Crawler le site et extraire les URLs des vidéos
print("Début du crawl du site...")
video_urls = crawl_website(base_url)

# Télécharger les vidéos si leur durée est inférieure ou égale à 8 secondes
if video_urls:
    print(f"{len(video_urls)} vidéos trouvées. Début du téléchargement...")
    for video_url in video_urls:
        duration = get_video_duration(video_url)
        if duration is not None:
            if duration <= 8:
                download_video(video_url)
            else:
                print(f"Vidéo ignorée (durée : {duration:.2f} secondes) : {video_url}")
        else:
            print(f"Impossible de déterminer la durée de la vidéo : {video_url}")
else:
    print("Aucune vidéo trouvée.")