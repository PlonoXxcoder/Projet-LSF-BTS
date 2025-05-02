# config.py

import os

# -----------------------------------------------------------------------------
# Chemins de Base et Configuration Principale
# -----------------------------------------------------------------------------
# Répertoire racine du projet (où se trouve ce fichier config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paramètre Principal : Choisir entre CNN Features et Keypoints ---
# True: Utilise les features CNN extraites comme entrée du LSTM (mode actif).
# False: Utilise les keypoints MediaPipe comme entrée du LSTM.
USE_CNN_FEATURES = True # MODIFIEZ CECI POUR CHANGER DE MODE

# --- Paramètres Généraux (Communs aux deux approches) ---
# Longueur fixe des séquences temporelles pour le modèle LSTM
FIXED_LENGTH = 30

# --- Noms de Fichiers Essentiels / Sortie ---
# Fichier texte contenant les labels (un par ligne, utilisé pour le vocabulaire)
VOCABULARY_FILE = "vocabulaire.txt"
# Nom du fichier CSV pour enregistrer les prédictions faites par CaptureVideo.py
PREDICTION_CSV_FILENAME = "live_predictions.csv"

# --- Dossiers Principaux ---
# Dossier contenant les vidéos sources pour l'entraînement/validation/test
VIDEOS_SOURCE_DIR = "video"
# Dossier pour sauvegarder les modèles Keras entraînés (.keras)
MODEL_DIR = "models"
# Dossier pour les logs (TensorBoard, CSV de prédictions, autres logs)
LOG_DIR = "logs"
# Crée les dossiers principaux s'ils n'existent pas (sécurité)
os.makedirs(os.path.join(BASE_DIR, VIDEOS_SOURCE_DIR), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, MODEL_DIR), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, LOG_DIR), exist_ok=True)

# -----------------------------------------------------------------------------
# Section: Configuration pour l'Approche CNN + LSTM
# (Utilisée uniquement si USE_CNN_FEATURES = True)
# -----------------------------------------------------------------------------
if USE_CNN_FEATURES:
    # --- Choix du Modèle CNN Pré-entraîné ---
    CNN_MODEL_CHOICE = 'MobileNetV2' # Ex: 'MobileNetV2', 'ResNet50', 'EfficientNetB0'

    # --- Paramètres du CNN ---
    CNN_INPUT_SHAPE = (224, 224, 3) # (Hauteur, Largeur, Canaux)

    # --- Dossier pour Sauvegarder les Caractéristiques CNN extraites (.npy) ---
    CNN_DATA_SAVE_DIR = "data_cnn_features"
    os.makedirs(os.path.join(BASE_DIR, CNN_DATA_SAVE_DIR), exist_ok=True)

    # --- Taille du Vecteur de Caractéristiques CNN (Calculé/Défini) ---
    if CNN_MODEL_CHOICE == 'MobileNetV2':
        CNN_FEATURE_VECTOR_SIZE = 1280
    elif CNN_MODEL_CHOICE == 'ResNet50':
        CNN_FEATURE_VECTOR_SIZE = 2048
    elif CNN_MODEL_CHOICE == 'EfficientNetB0':
        CNN_FEATURE_VECTOR_SIZE = 1280
    elif CNN_MODEL_CHOICE == 'InceptionV3':
        CNN_FEATURE_VECTOR_SIZE = 2048
    else:
        print(f"AVERTISSEMENT: CNN_FEATURE_VECTOR_SIZE non défini pour {CNN_MODEL_CHOICE}. Défaut à 1024 (INCORRECT).")
        CNN_FEATURE_VECTOR_SIZE = 1024

    # --- Noms de fichiers spécifiques à CNN+LSTM ---
    CNN_LSTM_MODEL_FILENAME = f"model_{CNN_MODEL_CHOICE.lower()}_lstm.keras"
    CNN_LSTM_BEST_MODEL_CHECKPOINT_FILENAME = f"best_model_{CNN_MODEL_CHOICE.lower()}_lstm_checkpoint.h5"
    CNN_LSTM_COMBINED_DATA_FEATURES_FILENAME = "combined_cnn_features.npy"
    CNN_LSTM_COMBINED_DATA_LABELS_FILENAME = "combined_cnn_labels.npy"


# -----------------------------------------------------------------------------
# Section: Configuration pour l'Approche Keypoints + LSTM
# (Utilisée uniquement si USE_CNN_FEATURES = False)
# -----------------------------------------------------------------------------
if not USE_CNN_FEATURES:
    # --- Dossier pour Sauvegarder les Keypoints extraits (.npy) ---
    DATA_SAVE_DIR = "data_keypoints"
    os.makedirs(os.path.join(BASE_DIR, DATA_SAVE_DIR), exist_ok=True)
    CAPTURE_KEYPOINTS_SAVE_DIR = "extracted_keypoints_capture"
    os.makedirs(os.path.join(BASE_DIR, CAPTURE_KEYPOINTS_SAVE_DIR), exist_ok=True)

    # --- Calcul de la Dimension des Features par Frame (Keypoints) ---
    # !! IMPORTANT !! : Mettez à jour ce calcul si vous activez ce mode.
    FEATURES_PER_FRAME = 1662 # Exemple basé sur Pose(33*4) + Face(468*3) + 2 Mains(2*21*3)
    print(f"INFO (Mode Keypoints Actif ou Config Lues): FEATURES_PER_FRAME = {FEATURES_PER_FRAME}.")

    # --- Noms de fichiers spécifiques à Keypoints+LSTM ---
    MODEL_FILENAME = "model_keypoints_lstm.keras"
    BEST_MODEL_CHECKPOINT_FILENAME = "best_model_keypoints_lstm_checkpoint.h5"
    COMBINED_DATA_KEYPOINTS_FILENAME = "combined_keypoints_features.npy"
    COMBINED_DATA_LABELS_FILENAME = "combined_keypoints_labels.npy"


# -----------------------------------------------------------------------------
# Sélection des Paramètres Actifs (basé sur USE_CNN_FEATURES)
# Les scripts devraient utiliser ces variables `ACTIVE_`
# -----------------------------------------------------------------------------
if USE_CNN_FEATURES:
    ACTIVE_FEATURE_DIM = CNN_FEATURE_VECTOR_SIZE
    ACTIVE_DATA_SAVE_DIR = os.path.join(BASE_DIR, CNN_DATA_SAVE_DIR)
    ACTIVE_MODEL_FILENAME = CNN_LSTM_MODEL_FILENAME # Juste le nom de fichier
    ACTIVE_BEST_MODEL_CHECKPOINT_FILENAME = CNN_LSTM_BEST_MODEL_CHECKPOINT_FILENAME # Juste le nom
    # Noms de fichiers pour données combinées (pas de chemin ici, juste les noms)
    ACTIVE_COMBINED_DATA_FEATURES_FILENAME = CNN_LSTM_COMBINED_DATA_FEATURES_FILENAME
    ACTIVE_COMBINED_DATA_LABELS_FILENAME = CNN_LSTM_COMBINED_DATA_LABELS_FILENAME
    FEATURE_FILE_SUFFIX = "_cnn_features.npy" # Suffixe pour load_data
else:
    ACTIVE_FEATURE_DIM = FEATURES_PER_FRAME
    ACTIVE_DATA_SAVE_DIR = os.path.join(BASE_DIR, DATA_SAVE_DIR)
    ACTIVE_MODEL_FILENAME = MODEL_FILENAME # Juste le nom de fichier
    ACTIVE_BEST_MODEL_CHECKPOINT_FILENAME = BEST_MODEL_CHECKPOINT_FILENAME # Juste le nom
    # Noms de fichiers pour données combinées
    ACTIVE_COMBINED_DATA_FEATURES_FILENAME = COMBINED_DATA_KEYPOINTS_FILENAME
    ACTIVE_COMBINED_DATA_LABELS_FILENAME = COMBINED_DATA_LABELS_FILENAME
    FEATURE_FILE_SUFFIX = "_keypoints.npy" # Suffixe pour load_data


# --- Chemins Absolus Construits (pour la commodité) ---
# Chemin complet vers le fichier vocabulaire
VOCABULARY_PATH = os.path.join(BASE_DIR, VOCABULARY_FILE)
# Chemin complet vers le fichier CSV de prédictions
PREDICTION_CSV_PATH = os.path.join(BASE_DIR, LOG_DIR, PREDICTION_CSV_FILENAME)
# Chemins complets vers le modèle et le checkpoint actifs
ACTIVE_MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR, ACTIVE_MODEL_FILENAME)
ACTIVE_BEST_MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, MODEL_DIR, ACTIVE_BEST_MODEL_CHECKPOINT_FILENAME)
# Chemins complets vers les fichiers de données combinées actifs
ACTIVE_COMBINED_FEATURES_PATH = os.path.join(ACTIVE_DATA_SAVE_DIR, ACTIVE_COMBINED_DATA_FEATURES_FILENAME)
ACTIVE_COMBINED_LABELS_PATH = os.path.join(ACTIVE_DATA_SAVE_DIR, ACTIVE_COMBINED_DATA_LABELS_FILENAME)

# -----------------------------------------------------------------------------
# Paramètres d'Entraînement (Utilisés par Entrainement.py)
# -----------------------------------------------------------------------------
# --- Contrôle Général ---
TRAIN_EPOCHS_FINAL = 50          # Nombre max d'époques pour l'entraînement final
TRAIN_BATCH_SIZE = 32            # Taille du lot pour entraînement/évaluation
# --- Équilibrage ---
TRAIN_APPLY_SMOTE = False         # Activer SMOTE pour équilibrer le jeu d'entraînement (nécessite imbalanced-learn)
# --- Keras Tuner ---
TRAIN_USE_KERAS_TUNER = True     # Activer la recherche d'hyperparamètres avec Keras Tuner
TRAIN_EPOCHS_TUNING = 10         # Nombre max d'époques PAR essai du tuner (si TRAIN_USE_KERAS_TUNER=True)
TRAIN_TUNER_PROJECT_NAME = "SignLanguageTuner_CNN_LSTM" # Nom du projet pour les résultats du tuner
# --- Boucle de Ré-entraînement (Optionnelle, surtout utile si pas de tuner) ---
TRAIN_ACCURACY_THRESHOLD = 0.85  # Seuil d'accuracy sur le Test Set pour arrêter l'entraînement
TRAIN_MAX_RETRAIN_ATTEMPTS = 3   # Nombre max de tentatives si le seuil n'est pas atteint
# --- Callbacks ---
TRAIN_PATIENCE_EARLY_STOPPING = 15 # Patience pour EarlyStopping pendant l'entraînement final
TRAIN_PATIENCE_REDUCE_LR = 7     # Patience pour ReduceLROnPlateau
TRAIN_FACTOR_REDUCE_LR = 0.5     # Facteur de réduction du LR
TRAIN_MIN_LR = 0.00001           # Taux d'apprentissage minimal
# --- Split Données ---
TRAIN_TEST_SIZE = 0.15           # Proportion pour le Test Set (ex: 15%)
TRAIN_VALIDATION_SIZE = 0.15     # Proportion pour le Validation Set (ex: 15%) - Le reste sera pour Train
TRAIN_LEARNING_RATE = 0.001   
# -----------------------------------------------------------------------------
# Paramètres pour le Multiprocessing/Threading (Utilisé pour l'Extraction de Features)
# -----------------------------------------------------------------------------
# Pour CNN (GPU), 1 est plus sûr. Pour Keypoints (CPU), peut être augmenté.
PROCESS_MAX_WORKERS = 1 if USE_CNN_FEATURES else (os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1)

# -----------------------------------------------------------------------------
# Paramètres pour la Capture Vidéo en Temps Réel (Utilisés par CaptureVideo.py)
# -----------------------------------------------------------------------------
# --- Source et Dimensions ---
CAPTURE_SOURCE = 0               # 0 pour webcam, ou chemin/URL vidéo
CAPTURE_WIDTH = 640              # Largeur affichage/traitement souhaitée
CAPTURE_HEIGHT = 480             # Hauteur affichage/traitement souhaitée
CAPTURE_MAX_FRAME_WIDTH = 1280   # Largeur max de la frame traitée (None pour illimité)

# --- Répertoire spécifique pour les vidéos de test utilisées par CaptureVideo.py ---
CAPTURE_VIDEO_TEST_DIR = "video_test" # Chemin relatif au BASE_DIR

# --- Paramètres de Prédiction/Affichage ---
PREDICTION_THRESHOLD = 0.70      # Seuil de confiance MINIMUM pour considérer une prédiction
CAPTURE_CONF_THRESH_GREEN = 0.90 # Seuil pour affichage en VERT (très sûr)
CAPTURE_CONF_THRESH_YELLOW = 0.80 # Seuil pour affichage en JAUNE (assez sûr)
DISPLAY_PROBABILITIES = True     # Afficher les probabilités détaillées ?
CAPTURE_TOP_N = 3                # Nombre de prédictions (Top-N) à afficher

# --- Paramètres de Performance/Stabilité ---
CAPTURE_SMOOTHING_WINDOW_SIZE = 5 # Fenêtre pour lissage prédictions (1 = pas de lissage)
FRAMES_TO_SKIP = 0               # Nb frames à SAUTER entre traitements (0 = traiter tout)
CAPTURE_DEADLOCK_TIMEOUT = 10.0  # Timeout détection blocage (non utilisé dans le code actuel)


# -----------------------------------------------------------------------------
# Vérification et Affichage de la Configuration Active (Fonction Utile pour Debug)
# -----------------------------------------------------------------------------
def print_active_config():
    """Affiche un résumé de la configuration active dans la console."""
    print("\n" + "="*70)
    print("--- Configuration Globale et Active ---")
    print(f"Mode Actif: {'CNN + LSTM' if USE_CNN_FEATURES else 'Keypoints + LSTM'}")
    print(f"Répertoire Base Projet (BASE_DIR): {BASE_DIR}")
    print(f"Longueur Séquence LSTM (FIXED_LENGTH): {FIXED_LENGTH}")
    print(f"Fichier Vocabulaire (Chemin): {VOCABULARY_PATH}")
    print(f"Fichier CSV Prédictions (Chemin): {PREDICTION_CSV_PATH}")
    print("-" * 40)
    print("--- Chemins Actifs (Modèle/Données) ---")
    print(f"Dossier Données Traitées Actives: {ACTIVE_DATA_SAVE_DIR}")
    print(f"  -> Fichier Features Combinées: {ACTIVE_COMBINED_FEATURES_PATH}")
    print(f"  -> Fichier Labels Combinées: {ACTIVE_COMBINED_LABELS_PATH}")
    print(f"Fichier Modèle Actif: {ACTIVE_MODEL_PATH}")
    print(f"Fichier Checkpoint Actif: {ACTIVE_BEST_MODEL_CHECKPOINT_PATH}")
    print(f"Suffixe Fichier Features Individuel: {FEATURE_FILE_SUFFIX}")
    print("-" * 40)
    print("--- Dimensions & Modèles ---")
    print(f"Dimension Features Entrée LSTM (ACTIVE_FEATURE_DIM): {ACTIVE_FEATURE_DIM}")
    if USE_CNN_FEATURES:
        print(f"  Mode CNN Activé:")
        print(f"    Modèle CNN Base Choisi: {CNN_MODEL_CHOICE}")
        print(f"    Shape Entrée CNN Attendue: {CNN_INPUT_SHAPE}")
        print(f"    Taille Vecteur Features CNN: {CNN_FEATURE_VECTOR_SIZE}")
    else:
        print(f"  Mode Keypoints Activé:")
        print(f"    Taille Vecteur Features Keypoints: {FEATURES_PER_FRAME} (VÉRIFIEZ!)")
    print("-" * 40)
    print("--- Paramètres d'Entraînement (Entrainement.py) ---")
    print(f"Activer SMOTE: {TRAIN_APPLY_SMOTE}")
    print(f"Activer Keras Tuner: {TRAIN_USE_KERAS_TUNER}")
    print(f"  -> Epochs Max par essai Tuner: {TRAIN_EPOCHS_TUNING}")
    print(f"  -> Nom Projet Tuner: {TRAIN_TUNER_PROJECT_NAME}")
    print(f"Epochs Max Entraînement Final: {TRAIN_EPOCHS_FINAL}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}")
    print(f"Seuil Accuracy Objectif: {TRAIN_ACCURACY_THRESHOLD}")
    print(f"Max Tentatives Ré-entraînement: {TRAIN_MAX_RETRAIN_ATTEMPTS}")
    print(f"Proportion Test Set: {TRAIN_TEST_SIZE}")
    print(f"Proportion Validation Set: {TRAIN_VALIDATION_SIZE}")
    print(f"Patience Early Stopping: {TRAIN_PATIENCE_EARLY_STOPPING}")
    print(f"Patience Reduce LR: {TRAIN_PATIENCE_REDUCE_LR}")
    print("-" * 40)
    print("--- Paramètres Capture Vidéo (CaptureVideo.py) ---")
    print(f"Source Capture: {CAPTURE_SOURCE}")
    print(f"Largeur/Hauteur Capture Souhaitée: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"Largeur Max Frame Traitée: {CAPTURE_MAX_FRAME_WIDTH or 'Non limité'}")
    print(f"Dossier Vidéos Test Capture: {os.path.join(BASE_DIR, CAPTURE_VIDEO_TEST_DIR)}")
    print(f"Seuils Confiance (Min/Jaune/Vert): {PREDICTION_THRESHOLD}/{CAPTURE_CONF_THRESH_YELLOW}/{CAPTURE_CONF_THRESH_GREEN}")
    print(f"Afficher Top N Prédictions: {CAPTURE_TOP_N}")
    print(f"Fenêtre Lissage Prédictions: {CAPTURE_SMOOTHING_WINDOW_SIZE} (1 = aucun)")
    print(f"Afficher Probabilités Détaillées: {DISPLAY_PROBABILITIES}")
    print(f"Frames à sauter par traitement: {FRAMES_TO_SKIP} (0 = aucune)")
    # print(f"Timeout Deadlock (non utilisé): {CAPTURE_DEADLOCK_TIMEOUT}s")
    print("-" * 40)
    print("--- Autres Paramètres ---")
    print(f"Workers Max (Pour Extraction Features): {PROCESS_MAX_WORKERS}")
    print("="*70 + "\n")

# --- Exécution Optionnelle à l'Import ---
# Décommenter pour voir la config à chaque import de `config`.
# print_active_config()

# Fin du fichier config.py