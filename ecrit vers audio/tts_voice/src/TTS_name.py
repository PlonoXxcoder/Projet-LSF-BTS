# Python standard modules
import os
import requests
import base64
import re
import queue
import threading
import time
import tempfile
from json import load, dump # Ajout de dump
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys

# Downloaded modules
from gtts import gTTS
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
except ImportError:
    # ... (gestion de l'erreur pydub inchangée) ...
    PYDUB_AVAILABLE = False

# --- Étape 1: Mettre à jour l'Enum Voice ---
# Supposons que voice.py existe au même niveau ou dans le package
# Si ce n'est pas le cas, définissez l'Enum ici :
try:
    # Si vous aviez un fichier voice.py, il faudrait le mettre à jour comme ci-dessous
    # from .voice import Voice
    # Pour cet exemple, on définit directement ici :
    class Voice(Enum):
        FR_MALE = "fr_002"  # Voix masculine française existante
        # REMPLACEZ ces valeurs par les VRAIS identifiants de votre API pour les voix anglaises
        EN_MALE = "en_us_006"  # Exemple: Voix masculine anglaise (à adapter)
        EN_FEMALE = "en_us_001" # Exemple: Voix féminine anglaise (à adapter)
        # Ajoutez d'autres voix API ici si nécessaire

except ImportError:
     # Définition directe si l'import échoue (comme avant, mais mise à jour)
    class Voice(Enum):
        FR_MALE = "fr_002"
        # REMPLACEZ ces valeurs par les VRAIS identifiants de votre API pour les voix anglaises
        EN_MALE = "en_us_006"
        EN_FEMALE = "en_us_001"

# Créer un mapping pour faciliter la recherche de l'enum par clé string
# On exclut fr_female car elle utilise gTTS
api_voice_keys = {v.name.lower(): v for v in Voice} # ex: {"fr_male": Voice.FR_MALE, "en_male": Voice.EN_MALE, ...}
valid_voice_keys = list(api_voice_keys.keys()) + ["fr_female"] # Toutes les clés valides


# --- Fonctions utilitaires (inchangées) ---
def _load_endpoints() -> List[Dict[str, str]]:
    # ... (code inchangé) ...
    script_dir = os.path.dirname(__file__)
    json_file_path = os.path.join(script_dir, 'data', 'config.json')
    if not os.path.exists(json_file_path):
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
    # ... (code inchangé) ...
    merged_chunks: List[str] = []
    separated_chunks: List[str] = re.findall(r'.*?[.,!?:;-]|.+', text)
    character_limit: int = 300

    processed_chunks: List[str] = []
    for chunk in separated_chunks:
        if len(chunk.encode("utf-8")) > character_limit:
             sub_chunks = re.findall(r'.*?[ ]|.+', chunk)
             processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(chunk)

    current_chunk: str = ""
    for separated_chunk in processed_chunks:
        if len((current_chunk + separated_chunk).encode("utf-8")) <= character_limit:
            current_chunk += separated_chunk
        else:
            if current_chunk:
                 merged_chunks.append(current_chunk)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                print(f"Warning: Segment too long even after splitting: {separated_chunk[:50]}...")
                merged_chunks.append(separated_chunk)
                current_chunk = ""

    if current_chunk:
        merged_chunks.append(current_chunk)

    return [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]

def _fetch_audio_bytes_from_api( # Renommée pour la clarté
    endpoint: Dict[str, str],
    text_chunk: str,
    voice_id: str # Prend l'ID string directement
) -> Optional[str]:
    """Fetch a single audio chunk (base64 encoded string) from the API."""
    try:
        # Utilise voice_id reçu en argument
        print(f"  [API TTS] Sending chunk to {endpoint.get('url', 'N/A')} (Voice: {voice_id}): {text_chunk[:30]}...")
        response = requests.post(endpoint["url"], json={"text": text_chunk, "voice": voice_id}, timeout=15)
        response.raise_for_status()
        json_response = response.json()
        if endpoint["response"] in json_response:
             print(f"  [API TTS] Received chunk data.")
             return json_response[endpoint["response"]]
        else:
             print(f"  [API TTS] Error: Key '{endpoint['response']}' not found in response: {json_response}")
             return None
    except requests.exceptions.Timeout:
        print(f"  [API TTS] Error: Request timed out for chunk: {text_chunk[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [API TTS] Error fetching audio chunk: {e}")
        return None
    except KeyError:
        print(f"  [API TTS] Error: Invalid endpoint configuration or API response structure.")
        return None
    except Exception as e:
        print(f"  [API TTS] An unexpected error occurred during fetch: {e}")
        return None

# --- Étape 3: Renommer et adapter generate_male_audio ---
def generate_api_audio(text: str, output_file_path: str, voice: Voice) -> bool:
    """Generates audio using the custom API for the specified voice and saves to file."""
    print(f"[API TTS] Generating audio for voice {voice.name} ({voice.value}): {text[:50]}...")
    try:
        _validate_args(text, voice) # Valide les arguments (fonction inchangée)
    except (TypeError, ValueError) as e:
        print(f"[API TTS] Error: Invalid arguments - {e}")
        return False

    endpoint_data = _load_endpoints()
    if not endpoint_data:
        print("[API TTS] Error: No endpoints loaded. Cannot generate audio.")
        return False

    for endpoint in endpoint_data:
        print(f"[API TTS] Trying endpoint: {endpoint.get('url', 'N/A')}")
        text_chunks: List[str] = _split_text(text)
        if not text_chunks:
            print("[API TTS] Error: Text resulted in empty chunks after splitting.")
            continue

        # --- Utilisation de Threads (inchangé mais appelle _fetch_audio_bytes_from_api) ---
        audio_chunks_b64: List[Optional[str]] = [None] * len(text_chunks)
        threads: List[threading.Thread] = []
        results: Dict[int, Optional[str]] = {}

        def thread_target(index: int, chunk: str):
            # Passe la valeur string de l'enum (l'ID de la voix)
            results[index] = _fetch_audio_bytes_from_api(endpoint, chunk, voice.value)

        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        # --- Fin Threads ---

        for i in range(len(text_chunks)):
            audio_chunks_b64[i] = results.get(i)

        # --- Traitement des résultats (inchangé) ---
        if all(chunk is not None for chunk in audio_chunks_b64):
            print("[API TTS] All chunks received successfully. Concatenating and decoding...")
            try:
                full_audio_b64 = "".join([chunk for chunk in audio_chunks_b64 if chunk is not None])
                audio_bytes = base64.b64decode(full_audio_b64)
                _save_audio_file(output_file_path, audio_bytes)
                print(f"[API TTS] Audio successfully saved to {output_file_path}")
                return True
            # ... (gestion des erreurs inchangée) ...
            except base64.binascii.Error as e:
                print(f"[API TTS] Error decoding base64 string: {e}")
            except IOError as e:
                print(f"[API TTS] Error saving audio file {output_file_path}: {e}")
                return False
            except Exception as e:
                print(f"[API TTS] Unexpected error during saving/decoding: {e}")
        else:
             print(f"[API TTS] Failed to fetch all audio chunks for endpoint {endpoint.get('url', 'N/A')}. Missing chunks: {[i for i, ch in enumerate(audio_chunks_b64) if ch is None]}")


    print("[API TTS] Error: Failed to generate audio using all available endpoints.")
    return False

def _save_audio_file(output_file_path: str, audio_bytes: bytes):
    # ... (code inchangé) ...
    try:
        with open(output_file_path, "wb") as file:
            file.write(audio_bytes)
        print(f"  [Util] Audio data written to {output_file_path}")
    except IOError as e:
        print(f"  [Util] Error writing audio file {output_file_path}: {e}")
        raise

def _validate_args(text: str, voice: Voice):
    # ... (code inchangé - valide toujours un membre de l'Enum Voice) ...
    if not isinstance(voice, Voice):
        raise TypeError(f"'voice' must be of type Voice, got {type(voice)}")
    if not text or text.isspace():
        raise ValueError("text must not be empty or whitespace")

# --- Fin: Adaptation de votre code original ---

# --- Début: Intégration gTTS et Système de File d'attente ---

# File d'attente pour les requêtes TTS
tts_queue: queue.Queue = queue.Queue()
stop_worker = threading.Event()
# MALE_VOICE_ID n'est plus nécessaire globalement ici

# Vitesse de lecture globale (inchangé)
playback_speed: float = 1.0

def generate_female_audio(text: str, output_file_path: str) -> bool:
    """Generates female voice audio using gTTS and saves to file."""
    # Renommée pour clarifier que c'est gTTS / Français
    print(f"[FR Female TTS - gTTS] Generating audio for: {text[:50]}...")
    try:
        tts = gTTS(text=text, lang='fr', slow=False)
        tts.save(output_file_path)
        print(f"[FR Female TTS - gTTS] Audio successfully saved to {output_file_path}")
        return True
    except Exception as e:
        print(f"[FR Female TTS - gTTS] Error generating or saving audio: {e}")
        return False

# --- Fonctions de lecture audio (speed_change, play_audio_file) ---
# ... (code inchangé, incluant la correction pour "Weird sample rates") ...
def speed_change(sound: AudioSegment, speed: float = 1.0) -> AudioSegment:
    """Changes the speed of an AudioSegment. Speed > 1.0 is faster, < 1.0 is slower."""
    if speed == 1.0:
        return sound
    print(f"  [Playback] Applying speed factor: {speed:.2f}")
    # Modification de frame_rate simple (change le pitch)
    new_frame_rate = int(sound.frame_rate * speed)
    # Empêche une frame_rate de 0 ou négative qui causerait une erreur
    if new_frame_rate <= 0:
        print(f"  [Playback] Warning: Calculated frame rate ({new_frame_rate}) is invalid. Using original rate.")
        return sound
    return sound._spawn(sound.raw_data, overrides={
        "frame_rate": new_frame_rate
    })

def play_audio_file(file_path: str):
    """Plays an audio file using pydub, applying the global playback speed."""
    global playback_speed

    if not PYDUB_AVAILABLE:
        print("[Playback] Pydub not available. Skipping playback.")
        return

    final_audio_to_play = None # Initialiser pour le bloc except

    try:
        print(f"[Playback] Loading {file_path}...")
        file_extension = os.path.splitext(file_path)[1].lower().strip('.')
        if not file_extension:
            file_extension = "mp3"
            print(f"[Playback] Warning: No file extension detected, assuming '{file_extension}'.")

        audio = AudioSegment.from_file(file_path, format=file_extension)

        audio_at_speed = speed_change(audio, playback_speed)

        target_frame_rate = audio.frame_rate if playback_speed == 1.0 else 44100

        if int(audio_at_speed.frame_rate) != target_frame_rate and playback_speed != 1.0:
            print(f"  [Playback] Resampling from {audio_at_speed.frame_rate} Hz to {target_frame_rate} Hz for compatibility...")
            final_audio_to_play = audio_at_speed.set_frame_rate(target_frame_rate)
        else:
             final_audio_to_play = audio_at_speed

        print(f"[Playback] Playing at {playback_speed:.2f}x speed (Sample Rate: {final_audio_to_play.frame_rate} Hz)...")
        pydub_play(final_audio_to_play)
        print(f"[Playback] Finished playing {file_path}.")

    except FileNotFoundError:
        print(f"[Playback] Error: File not found at {file_path}")
    except Exception as e:
        current_rate = final_audio_to_play.frame_rate if final_audio_to_play else "N/A"
        if "Weird sample rates" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Even after resampling, the sample rate {current_rate} Hz might not be supported by the backend.")
        # Gérer aussi le cas où ffmpeg/ffprobe ne sont pas trouvés PENDANT la lecture/conversion implicite
        elif "Cannot find ffprobe" in str(e) or "Cannot find ffmpeg" in str(e) or "[WinError 2]" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Error: Ensure 'ffmpeg'/'ffprobe' executables are found by pydub.")
             print(f"[Playback] Check your system PATH or explicit configuration if used.")
        # Gérer l'erreur de permission [Errno 13]
        elif "[Errno 13] Permission denied" in str(e):
             print(f"[Playback] Error playing sound file {file_path}: {e}")
             print(f"[Playback] Permission denied, possibly when creating/accessing a temporary WAV file for playback.")
             print(f"[Playback] Check antivirus or folder permissions for the Temp directory.")
        else:
            print(f"[Playback] Error playing sound file {file_path}: {e}")
            print("[Playback] Ensure 'ffmpeg' is installed and in your system's PATH.")
            print("[Playback] Also check if the audio file format is supported or if there are permission issues.")


# --- Étape 2 (partie worker): Modifier tts_worker ---
def tts_worker():
    """Worker thread that processes TTS requests from the queue."""
    print("[Worker] TTS Worker thread started.")
    while not stop_worker.is_set():
        try:
            # Récupère maintenant (texte, voice_key_string)
            text, voice_key = tts_queue.get(timeout=1.0)
            print(f"\n[Worker] Processing request: VoiceKey='{voice_key}', Text='{text[:50]}...'")

            success = False
            temp_file_path = None

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                    temp_file_path = tmp_audio_file.name
                    print(f"[Worker] Using temporary file: {temp_file_path}")

                # --- Logique de décision basée sur voice_key ---
                if voice_key == 'fr_female':
                    # Utiliser gTTS pour la voix féminine française
                    success = generate_female_audio(text, temp_file_path)
                elif voice_key in api_voice_keys:
                    # Utiliser l'API pour les autres voix (fr_male, en_male, en_female)
                    target_voice_enum = api_voice_keys[voice_key] # Trouve l'Enum correspondant
                    success = generate_api_audio(text, temp_file_path, target_voice_enum)
                else:
                    print(f"[Worker] Error: Unknown voice key '{voice_key}'")
                # --- Fin logique de décision ---

                if success and temp_file_path and os.path.exists(temp_file_path):
                    play_audio_file(temp_file_path)
                elif not success:
                    print(f"[Worker] Failed to generate audio for request.")

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"[Worker] Cleaned up temporary file: {temp_file_path}")
                    except OSError as e:
                        print(f"[Worker] Error removing temporary file {temp_file_path}: {e}")

            tts_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            # ... (gestion d'erreur du worker inchangée) ...
            print(f"[Worker] Unexpected error in worker loop: {e}")
            # Ajouter traceback pour plus de détails en cas d'erreur imprévue
            import traceback
            traceback.print_exc()
            try:
                tts_queue.task_done()
            except ValueError:
                 pass
            time.sleep(1)

    print("[Worker] TTS Worker thread finished.")

# --- Étape 2 (partie API): Modifier speak ---
def speak(text: str, voice_key: str = 'fr_female'): # Prend voice_key string
    """Adds a text-to-speech request to the queue."""
    if not text or text.isspace():
        print("[API] Error: Text cannot be empty.")
        return

    voice_key = voice_key.lower() # Assurer la casse minuscule
    # Vérifier si la clé est valide (API ou gTTS)
    if voice_key not in valid_voice_keys:
        print(f"[API] Error: Invalid voice_key '{voice_key}'. Valid keys are: {', '.join(valid_voice_keys)}")
        return

    print(f"[API] Queuing request: VoiceKey={voice_key}, Text='{text[:50]}...'")
    # Met (texte, voice_key_string) dans la queue
    tts_queue.put((text, voice_key))

# --- Fin: Intégration gTTS et Système de File d'attente ---

# --- Début: Démarrage et Exemple d'utilisation ---

def get_playback_speed_from_user() -> float:
    # ... (code inchangé) ...
    default_speed = 1.0
    while True:
        try:
            speed_str = input(f"Entrez la vitesse de lecture (ex: 1.0 pour normal, 1.5 pour rapide, 0.8 pour lent) [Entrée pour {default_speed}]: ")
            if not speed_str:
                return default_speed
            speed = float(speed_str)
            if speed <= 0:
                print("La vitesse doit être un nombre positif.")
            else:
                return speed
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre (ex: 1.0, 1.2, 0.9).")
        except EOFError:
            print("\nEntrée annulée. Utilisation de la vitesse par défaut.")
            return default_speed

# --- Étape 4: Adapter run_example ---
def run_example():
    print("--- TTS System Ready ---")
    print(f"Vitesse de lecture réglée à : {playback_speed:.2f}x")
    # Met à jour l'instruction pour l'utilisateur
    print("Entrez le texte à synthétiser. Utilisez le préfixe de voix suivi de ':'")
    print(f"Préfixes valides: {', '.join(valid_voice_keys)}")
    print("Exemple: 'en_male: Hello world!' ou 'fr_female: Bonjour le monde!'")
    print("Si aucun préfixe n'est donné, 'fr_female' sera utilisé.")
    print("Tapez 'quit' ou 'exit' pour arrêter.")

    default_voice_key = 'fr_female' # Voix par défaut si pas de préfixe

    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['quit', 'exit']:
                break

            if ':' in user_input:
                parts = user_input.split(':', 1)
                v_key_input = parts[0].strip().lower() # Clé de voix entrée par l'utilisateur
                text_to_speak = parts[1].strip()

                # Vérifier si la clé entrée est valide
                if v_key_input in valid_voice_keys:
                    speak(text_to_speak, v_key_input)
                else:
                    print(f"Clé de voix invalide '{v_key_input}'. Utilisation de la voix par défaut '{default_voice_key}'.")
                    print(f"Clés valides: {', '.join(valid_voice_keys)}")
                    if text_to_speak: # S'assurer qu'il y a du texte même si la clé est invalide
                        speak(text_to_speak, default_voice_key)
                    else:
                        print("Aucun texte à synthétiser.")

            else:
                 # Pas de préfixe, utiliser la voix par défaut
                 text_to_speak = user_input.strip()
                 if text_to_speak:
                    print(f"Aucun préfixe détecté. Utilisation de la voix par défaut '{default_voice_key}'.")
                    speak(text_to_speak, default_voice_key)
                 # else: ne rien faire si l'entrée est vide

        except EOFError:
             break
        except KeyboardInterrupt:
             break

    print("\n[Main] Arrêt du système...")
    stop_worker.set()
    # Pas besoin de .join() si daemon=True, mais s'assurer que la queue est vide peut être utile
    # print("[Main] Waiting for queue to empty...")
    # tts_queue.join() # Attend que tous les éléments soient traités
    print("[Main] Programme terminé.")


if __name__ == "__main__":
    # --- Vérification et création config (inchangé) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    config_file = os.path.join(data_dir, 'config.json')
    # ... (le reste de la création/vérification de config.json est inchangé) ...
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Répertoire créé : {data_dir}")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire {data_dir}: {e}")
            sys.exit(1)
    if not os.path.exists(config_file):
        print(f"Attention : config.json non trouvé à {config_file}. Création d'un fichier par défaut.")
        print("Veuillez éditer 'data/config.json' avec vos vrais endpoints API.")
        # Mettre à jour les exemples si nécessaire, la structure reste la même
        dummy_config = [
            {"url": "https://tiktok-tts.weilnet.workers.dev/api/generation", "response": "data"},
            {"url": "https://example.com/api/tts", "response": "audio_base64"} # Exemple
        ]
        try:
            # import json # Déjà importé plus haut avec dump
            with open(config_file, 'w') as f:
                dump(dummy_config, f, indent=4)
        except Exception as e:
             print(f"Erreur lors de la création du fichier config.json par défaut : {e}")
             sys.exit(1)

    # --- Demander la vitesse à l'utilisateur (inchangé) ---
    playback_speed = get_playback_speed_from_user()

    # --- Démarrer le thread worker (inchangé) ---
    if not PYDUB_AVAILABLE:
         print("\nATTENTION : pydub non disponible, la lecture audio sera désactivée.")

    print("[Main] Démarrage du worker TTS...")
    tts_thread = threading.Thread(target=tts_worker, daemon=True) # daemon=True permet de quitter même si le thread tourne
    tts_thread.start()

    # --- Lancer l'exemple interactif ---
    run_example()

# --- Fin: Démarrage et Exemple d'utilisation ---