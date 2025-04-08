# Python standard modules
import os
import requests
import base64
import re
import queue
import threading
import time
import tempfile
from json import load
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Downloaded modules
from playsound import playsound
from gtts import gTTS # Importation de gTTS

# --- Début: Adaptation de votre code original ---

# Supposons que voice.py existe au même niveau ou dans le package
# Si ce n'est pas le cas, définissez l'Enum ici :
try:
    from .voice import Voice # Si c'est un package
except ImportError:
    # Ou définissez-le directement si ce n'est pas un package
    class Voice(Enum):
        MAN_DEFAULT = "fr_002" # REMPLACEZ par la vraie valeur attendue par votre API

# Fonctions utilitaires de votre code (légèrement adaptées si nécessaire)
def _load_endpoints() -> List[Dict[str, str]]:
    """Load endpoint configurations from a JSON file."""
    script_dir = os.path.dirname(__file__)
    # Ajustez le chemin si nécessaire
    json_file_path = os.path.join(script_dir, 'data', 'config.json')
    if not os.path.exists(json_file_path):
        # Fallback si 'data' n'est pas au même niveau que le script mais un niveau au-dessus
        json_file_path_alt = os.path.join(script_dir, '../data', 'config.json')
        if os.path.exists(json_file_path_alt):
             json_file_path = json_file_path_alt
        else:
             raise FileNotFoundError(f"Cannot find config.json in {os.path.join(script_dir, 'data')} or {os.path.join(script_dir, '../data')}")

    try:
        with open(json_file_path, 'r') as file:
            return load(file)
    except Exception as e:
        print(f"Error loading endpoints from {json_file_path}: {e}")
        return []

def _split_text(text: str) -> List[str]:
    """Split text into chunks suitable for the API (e.g., ~300 chars)."""
    # Votre logique de découpage existante
    merged_chunks: List[str] = []
    # Correction: Utiliser re.escape sur les délimiteurs si nécessaire ou simplifier
    # Simplification : découpe brutale aux ponctuations/espaces si la limite est dépassée
    # (Votre logique originale est conservée ici, mais peut nécessiter des ajustements)
    separated_chunks: List[str] = re.findall(r'.*?[.,!?:;-]|.+', text)
    character_limit: int = 300 # Limite par morceau pour l'API homme

    processed_chunks: List[str] = []
    for chunk in separated_chunks:
        # Mesurer en bytes peut être plus précis si l'API a une limite en bytes
        if len(chunk.encode("utf-8")) > character_limit:
             # Découper plus finement les morceaux trop longs (simple découpe par espace ici)
             sub_chunks = re.findall(r'.*?[ ]|.+', chunk)
             processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(chunk)

    # Regrouper pour ne pas dépasser la limite
    current_chunk: str = ""
    for separated_chunk in processed_chunks:
        # Vérifier la longueur *avant* d'ajouter pour éviter de dépasser
        if len((current_chunk + separated_chunk).encode("utf-8")) <= character_limit:
            current_chunk += separated_chunk
        else:
            # Si l'ajout dépasse, sauvegarder le morceau courant (s'il n'est pas vide)
            if current_chunk:
                 merged_chunks.append(current_chunk)
            # Commencer un nouveau morceau (gérer le cas où le nouveau morceau lui-même est trop long - normalement géré par le découpage précédent)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                # Ce cas est problématique - le mot/segment initial est déjà trop long
                # Pourrait nécessiter une découpe encore plus agressive ou lever une erreur
                print(f"Warning: Segment too long even after splitting: {separated_chunk[:50]}...")
                # On l'ajoute quand même pour ne pas perdre de texte, l'API pourrait échouer
                merged_chunks.append(separated_chunk)
                current_chunk = "" # Réinitialiser

    # Ajouter le dernier morceau s'il existe
    if current_chunk:
        merged_chunks.append(current_chunk)

    # Filtrer les morceaux vides qui pourraient résulter du découpage
    return [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]


def _fetch_male_audio_bytes(
    endpoint: Dict[str, str],
    text_chunk: str,
    voice: Voice
) -> Optional[str]:
    """Fetch a single audio chunk (base64 encoded string) from the male voice API."""
    try:
        print(f"  [Male TTS] Sending chunk to {endpoint.get('url', 'N/A')}: {text_chunk[:30]}...")
        response = requests.post(endpoint["url"], json={"text": text_chunk, "voice": voice.value}, timeout=15) # Ajout d'un timeout
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
        json_response = response.json()
        if endpoint["response"] in json_response:
             print(f"  [Male TTS] Received chunk data.")
             return json_response[endpoint["response"]]
        else:
             print(f"  [Male TTS] Error: Key '{endpoint['response']}' not found in response: {json_response}")
             return None
    except requests.exceptions.Timeout:
        print(f"  [Male TTS] Error: Request timed out for chunk: {text_chunk[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [Male TTS] Error fetching audio chunk: {e}")
        return None
    except KeyError:
        print(f"  [Male TTS] Error: Invalid endpoint configuration or API response structure.")
        return None
    except Exception as e:
        print(f"  [Male TTS] An unexpected error occurred during fetch: {e}")
        return None

def generate_male_audio(text: str, output_file_path: str, male_voice: Voice) -> bool:
    """Generates male voice audio using the custom API and saves to file."""
    print(f"[Male TTS] Generating audio for: {text[:50]}...")
    _validate_args(text, male_voice) # Valide les arguments

    endpoint_data = _load_endpoints()
    if not endpoint_data:
        print("[Male TTS] Error: No endpoints loaded. Cannot generate audio.")
        return False

    # Essayer les endpoints séquentiellement (comme dans le code original)
    for endpoint in endpoint_data:
        print(f"[Male TTS] Trying endpoint: {endpoint.get('url', 'N/A')}")
        text_chunks: List[str] = _split_text(text)
        if not text_chunks:
            print("[Male TTS] Error: Text resulted in empty chunks after splitting.")
            continue # Essayer l'endpoint suivant

        audio_chunks_b64: List[Optional[str]] = [None] * len(text_chunks)
        threads: List[threading.Thread] = []
        results: Dict[int, Optional[str]] = {}

        # Fonction cible pour les threads
        def thread_target(index: int, chunk: str):
            results[index] = _fetch_male_audio_bytes(endpoint, chunk, male_voice)

        # Démarrer les threads pour chaque morceau
        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk))
            threads.append(thread)
            thread.start()

        # Attendre la fin de tous les threads
        for thread in threads:
            thread.join()

        # Récupérer les résultats dans l'ordre
        for i in range(len(text_chunks)):
            audio_chunks_b64[i] = results.get(i)

        # Vérifier si tous les morceaux ont été récupérés avec succès
        if all(chunk is not None for chunk in audio_chunks_b64):
            print("[Male TTS] All chunks received successfully. Concatenating and decoding...")
            try:
                # Concaténer les chaînes base64 *avant* de décoder
                full_audio_b64 = "".join(audio_chunks_b64) # type: ignore
                audio_bytes = base64.b64decode(full_audio_b64)

                # Sauvegarder le fichier audio
                _save_audio_file(output_file_path, audio_bytes)
                print(f"[Male TTS] Audio successfully saved to {output_file_path}")
                return True # Succès, on arrête d'essayer les endpoints
            except base64.binascii.Error as e:
                print(f"[Male TTS] Error decoding base64 string: {e}")
                # Continuer pour essayer l'endpoint suivant
            except IOError as e:
                print(f"[Male TTS] Error saving audio file {output_file_path}: {e}")
                return False # Erreur d'écriture, probablement inutile d'essayer d'autres endpoints
            except Exception as e:
                print(f"[Male TTS] Unexpected error during saving/decoding: {e}")
                # Continuer pour essayer l'endpoint suivant

        else:
            print(f"[Male TTS] Failed to fetch all audio chunks for endpoint {endpoint.get('url', 'N/A')}. Missing chunks: {[i for i, ch in enumerate(audio_chunks_b64) if ch is None]}")
            # Continuer pour essayer l'endpoint suivant

    # Si on arrive ici, aucun endpoint n'a fonctionné
    print("[Male TTS] Error: Failed to generate audio using all available endpoints.")
    return False

def _save_audio_file(output_file_path: str, audio_bytes: bytes):
    """Write the audio bytes to a file."""
    try:
        # Supprimer l'ancien fichier s'il existe pour éviter les erreurs de playsound
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        with open(output_file_path, "wb") as file:
            file.write(audio_bytes)
        print(f"  [Util] Audio data written to {output_file_path}")
    except IOError as e:
        print(f"  [Util] Error writing audio file {output_file_path}: {e}")
        raise # Propage l'erreur pour que generate_male_audio puisse la gérer

def _validate_args(text: str, voice: Voice):
    """Validate the input arguments for male voice."""
    if not isinstance(voice, Voice):
        raise TypeError(f"'voice' must be of type Voice, got {type(voice)}")
    if not text or text.isspace():
        raise ValueError("text must not be empty or whitespace")

# --- Fin: Adaptation de votre code original ---

# --- Début: Intégration gTTS et Système de File d'attente ---

# File d'attente pour les requêtes TTS (texte, type_voix)
# Utilisation de Tuple[str, str] pour type_voix ('male' ou 'female')
tts_queue: queue.Queue = queue.Queue()

# Événement pour arrêter proprement le thread worker
stop_worker = threading.Event()

# Configuration de la voix masculine (à choisir parmi votre Enum Voice)
MALE_VOICE_ID = Voice.MAN_DEFAULT # Ou une autre voix de votre Enum

def generate_female_audio(text: str, output_file_path: str) -> bool:
    """Generates female voice audio using gTTS and saves to file."""
    print(f"[Female TTS] Generating audio for: {text[:50]}...")
    try:
        # Vous pouvez spécifier la langue, 'fr' pour français
        tts = gTTS(text=text, lang='fr', slow=False)
        tts.save(output_file_path)
        print(f"[Female TTS] Audio successfully saved to {output_file_path}")
        return True
    except Exception as e:
        print(f"[Female TTS] Error generating or saving audio: {e}")
        return False

def play_audio_file(file_path: str):
    """Plays an audio file using playsound and handles potential errors."""
    try:
        print(f"[Playback] Playing {file_path}...")
        playsound(file_path)
        print(f"[Playback] Finished playing {file_path}.")
    except Exception as e:
        # playsound peut lever diverses exceptions, notamment sur macOS ou Linux
        # si les dépendances (comme GStreamer) ne sont pas correctes.
        print(f"[Playback] Error playing sound file {file_path}: {e}")
        print("[Playback] Ensure you have the necessary backend for playsound (e.g., pygobject on Linux, AppKit on macOS).")

def tts_worker():
    """Worker thread that processes TTS requests from the queue."""
    print("[Worker] TTS Worker thread started.")
    while not stop_worker.is_set():
        try:
            # Attendre une tâche avec un timeout pour pouvoir vérifier stop_worker
            text, voice_type = tts_queue.get(timeout=1.0)
            print(f"\n[Worker] Processing request: Type={voice_type}, Text='{text[:50]}...'")

            success = False
            # Utiliser un fichier temporaire pour la sortie audio
            # delete=False est important car playsound a besoin du fichier existant
            # Nous le supprimerons manuellement après la lecture.
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                temp_file_path = tmp_audio_file.name
                print(f"[Worker] Using temporary file: {temp_file_path}")

            try:
                if voice_type == 'male':
                    success = generate_male_audio(text, temp_file_path, MALE_VOICE_ID)
                elif voice_type == 'female':
                    success = generate_female_audio(text, temp_file_path)
                else:
                    print(f"[Worker] Error: Unknown voice type '{voice_type}'")

                if success:
                    # Jouer l'audio généré seulement si la génération a réussi
                    play_audio_file(temp_file_path)
                else:
                    print(f"[Worker] Failed to generate audio for request.")

            finally:
                # Nettoyer le fichier temporaire après utilisation (ou tentative)
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"[Worker] Cleaned up temporary file: {temp_file_path}")
                    except OSError as e:
                        print(f"[Worker] Error removing temporary file {temp_file_path}: {e}")

            tts_queue.task_done() # Indiquer que la tâche est terminée

        except queue.Empty:
            # Timeout atteint, la file est vide, on continue la boucle pour vérifier stop_worker
            continue
        except Exception as e:
            print(f"[Worker] Unexpected error in worker loop: {e}")
            # Il pourrait être judicieux de marquer la tâche comme terminée même en cas d'erreur
            # pour éviter de bloquer potentiellement `queue.join()` si utilisé ailleurs.
            try:
                tts_queue.task_done()
            except ValueError: # Si task_done() est appelé alors qu'il n'y a pas de tâche active (peu probable ici)
                 pass
            time.sleep(1) # Petite pause en cas d'erreur répétée

    print("[Worker] TTS Worker thread finished.")

def speak(text: str, voice_type: str = 'female'):
    """Adds a text-to-speech request to the queue."""
    if not text or text.isspace():
        print("[API] Error: Text cannot be empty.")
        return

    voice_type = voice_type.lower()
    if voice_type not in ['male', 'female']:
        print(f"[API] Error: Invalid voice_type '{voice_type}'. Use 'male' or 'female'.")
        return

    print(f"[API] Queuing request: Type={voice_type}, Text='{text[:50]}...'")
    tts_queue.put((text, voice_type))

# --- Fin: Intégration gTTS et Système de File d'attente ---

# --- Début: Démarrage et Exemple d'utilisation ---

# Démarrer le thread worker
tts_thread = threading.Thread(target=tts_worker, daemon=True) # daemon=True permet au programme de quitter même si le thread tourne
tts_thread.start()

def run_example():
    print("--- TTS System Ready ---")
    print("Enter text to speak. Type 'male: Your text' or 'female: Your text'.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['quit', 'exit']:
                break

            if ':' in user_input:
                parts = user_input.split(':', 1)
                v_type = parts[0].strip().lower()
                text_to_speak = parts[1].strip()
                if v_type in ['male', 'female']:
                    speak(text_to_speak, v_type)
                else:
                    print(f"Invalid voice type '{v_type}'. Using default (female).")
                    speak(text_to_speak, 'female') # Ou 'male' si vous préférez
            else:
                # Par défaut, utiliser la voix féminine si non spécifié
                 speak(user_input.strip(), 'female')

        except EOFError: # Gérer Ctrl+D
             break
        except KeyboardInterrupt: # Gérer Ctrl+C
             break

    print("\n[Main] Shutting down...")
    # Optionnel : Attendre que la file se vide avant de quitter
    # print("[Main] Waiting for pending tasks to complete...")
    # tts_queue.join() # Décommenter si vous voulez attendre la fin des tâches en cours
    # print("[Main] All tasks completed.")

    # Signaler au worker de s'arrêter (si non daemon, ou pour nettoyage plus propre)
    # stop_worker.set()
    # tts_thread.join() # Attendre la fin du thread worker

    print("[Main] Program finished.")


if __name__ == "__main__":
    # Assurez-vous que le dossier 'data' et 'config.json' existent
    # Création minimale si absents pour éviter une erreur au démarrage
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data')
    config_file = os.path.join(data_dir, 'config.json')
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
        except OSError as e:
            print(f"Error creating directory {data_dir}: {e}")
    if not os.path.exists(config_file):
        print(f"Warning: config.json not found at {config_file}. Creating a dummy file.")
        print("Please edit 'data/config.json' with your actual API endpoint(s) for the male voice.")
        # Crée un fichier config.json minimal pour éviter l'erreur FileNotFoundError
        # Vous DEVEZ le remplir avec vos vrais endpoints !
        dummy_config = [
    {
        "url": "https://tiktok-tts.weilnet.workers.dev/api/generation",
        "response": "data"
    },
    {
        "url": "https://gesserit.co/api/tiktok-tts",
        "response": "base64"
    }
    ]
        try:
            import json
            with open(config_file, 'w') as f:
                json.dump(dummy_config, f, indent=4)
        except Exception as e:
             print(f"Error creating dummy config.json: {e}")

    # Lancer l'exemple interactif
    run_example()

# --- Fin: Démarrage et Exemple d'utilisation ---