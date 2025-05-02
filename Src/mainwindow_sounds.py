# --- PyQt Imports ---
import sys
print("DEBUG: Importing sys")
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale, # type: ignore
                            QMetaObject, QObject, QPoint, QPointF, QRect,
                            QSize, QTime, QUrl, Qt, Signal, QThread, Slot, QTimer)
print("DEBUG: Imported QtCore")
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, # type: ignore
                         QFont, QFontDatabase, QGradient, QIcon,
                         QImage, QKeySequence, QLinearGradient, QPainter,
                         QPalette, QPixmap, QRadialGradient, QTransform, QTextCursor)
print("DEBUG: Imported QtGui")
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QFrame,  # type: ignore
                             QGridLayout, QHBoxLayout, QLabel, QLayout,
                             QMainWindow, QMenuBar, QPushButton, QSizePolicy,
                             QSpacerItem, QStatusBar, QTextEdit, QVBoxLayout,
                             QWidget, QMessageBox, QButtonGroup, QComboBox) # Added QComboBox
print("DEBUG: Imported QtWidgets")

# --- Imports pour le traitement vidéo ---
# ... (OpenCV, Mediapipe, TF/Keras imports remain the same) ...
print("DEBUG: Importing OpenCV...")
try:
    import cv2
    print("DEBUG: OpenCV imported successfully.")
except ImportError:
    print("ERREUR: Le module 'cv2' (OpenCV) n'est pas installé.")
    print("Veuillez l'installer avec : pip install opencv-python")
    sys.exit(1)

print("DEBUG: Importing Mediapipe...")
try:
    import mediapipe as mp
    print("DEBUG: Mediapipe imported successfully.")
except ImportError:
    print("WARNING: Le module 'mediapipe' n'est pas installé (pip install mediapipe). Hand detection disabled.")
    mp = None # Définir mp à None si l'import échoue

print("DEBUG: Importing NumPy and TensorFlow...")
import numpy as np
# Suppress TensorFlow INFO/WARNING messages (optional)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1: Filter INFO, 2: Filter INFO/WARNING, 3: Filter ALL
import tensorflow as tf
print(f"DEBUG: TensorFlow imported. Version: {tf.__version__}")
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
print("DEBUG: Keras imports complete.")

import os
import logging
import time
from collections import deque, Counter
# import traceback # Keep commented unless needed for deep debugging

print("DEBUG: Standard library imports complete.")

# --- Import Config ---
print("DEBUG: Importing config...")
try:
    import config # Assurez-vous que config.py est dans le même dossier ou PYTHONPATH
    print("DEBUG: config imported successfully.")
except ImportError:
    print("ERREUR: Impossible d'importer config.py. Assurez-vous qu'il existe et qu'il est accessible.")
    sys.exit(1)
except Exception as e:
    print(f"ERREUR: Problème lors de l'import de config.py: {e}")
    sys.exit(1)

# --- TTS Imports ---
print("DEBUG: Importing TTS components...")
import queue
import threading
import tempfile
import base64
import requests
import re
from json import load as json_load, dump as json_dump
from typing import Dict, List, Optional, Tuple

# Import TTS library components (assuming tts_library folder structure)
try:
    from tts_library.src.voice import Voice
    # We won't directly use tts_library.tts function here, but reuse its logic within the worker
    print("DEBUG: TTS Voice enum imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR: Cannot import from tts_library ({e}). Ensure the folder structure is correct.")
    print("Expected structure: mainwindow.py, config.py, data/, tts_library/(__init__.py, src/(text_to_speech.py, voice.py))")
    sys.exit(1)

# Import TTS processing libraries
try:
    from gtts import gTTS
    print("DEBUG: gTTS imported.")
except ImportError:
    print("WARNING: gTTS not found (pip install gTTS). French Female voice ('fr_female') disabled.")
    gTTS = None

try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    PYDUB_AVAILABLE = True
    print("DEBUG: pydub imported.")
except ImportError:
    print("WARNING: pydub not found (pip install pydub). Audio playback disabled.")
    print("Ensure you also have FFmpeg installed and in your PATH for pydub.")
    PYDUB_AVAILABLE = False
    AudioSegment = None
    pydub_play = None
except Exception as e_pydub:
    print(f"ERROR importing or using pydub: {e_pydub}")
    print("Ensure you also have FFmpeg installed and in your PATH for pydub.")
    PYDUB_AVAILABLE = False
    AudioSegment = None
    pydub_play = None


# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
print("DEBUG: Logging configured.")

# --- Colors Class ---
# ... (Colors class remains the same) ...
class Colors:
    CV_BLUE = (255, 0, 0)
    CV_GREEN = (0, 255, 0)
    CV_RED = (0, 0, 255)
    CV_WHITE = (255, 255, 255)
    CV_YELLOW = (0, 255, 255)
    CV_LIGHT_GREY = (211, 211, 211) # Added for less important text

    try:
        MP_HAND_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
        MP_CONNECTION_DRAWING_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2) if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils') else None
    except AttributeError as e_colors_mp:
        print(f"WARN: Exception defining MediaPipe drawing specs (module/submodule loaded?): {e_colors_mp}")
        MP_HAND_DRAWING_SPEC = None
        MP_CONNECTION_DRAWING_SPEC = None
    except Exception as e_colors:
        print(f"ERREUR: Exception during Colors class definition (MediaPipe part): {e_colors}")
        MP_HAND_DRAWING_SPEC = None
        MP_CONNECTION_DRAWING_SPEC = None

# --- VideoThread Class ---
# ... (VideoThread class remains mostly the same, no TTS logic needed here) ...
class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    prediction_ready = Signal(str)
    top_n_ready = Signal(list)
    models_loaded = Signal(bool)
    error_occurred = Signal(str)
    hands_detected_signal = Signal(bool) # Keep this signal

    def __init__(self, parent=None):
        super().__init__(parent)
        print("DEBUG (VideoThread.__init__): Initializing...")
        self._running = False
        try:
            # Load config parameters
            self.MODEL_PATH = os.path.join(config.BASE_DIR, config.MODEL_DIR, config.ACTIVE_MODEL_FILENAME)
            self.VOCABULARY_PATH = config.VOCABULARY_PATH
            self.FIXED_LENGTH = config.FIXED_LENGTH
            self.FEATURE_DIM = config.ACTIVE_FEATURE_DIM
            self.CNN_MODEL_CHOICE = config.CNN_MODEL_CHOICE
            self.CNN_INPUT_SHAPE = config.CNN_INPUT_SHAPE
            self.CAPTURE_SOURCE = config.CAPTURE_SOURCE
            self.FRAMES_TO_SKIP = config.FRAMES_TO_SKIP
            self.PREDICTION_THRESHOLD = config.PREDICTION_THRESHOLD
            self.SMOOTHING_WINDOW_SIZE = config.CAPTURE_SMOOTHING_WINDOW_SIZE
            self.TOP_N = config.CAPTURE_TOP_N
            self.MAX_FRAME_WIDTH = config.CAPTURE_MAX_FRAME_WIDTH
            self.MIN_HAND_DETECTION_CONFIDENCE = getattr(config, 'MIN_HAND_DETECTION_CONFIDENCE', 0.5)
            self.MIN_HAND_TRACKING_CONFIDENCE = getattr(config, 'MIN_HAND_TRACKING_CONFIDENCE', 0.5)
            self.MAX_HANDS = getattr(config, 'MAX_HANDS', 2)
            self.ui_TOP_N = config.CAPTURE_TOP_N # Make sure this exists in config.py
            print("DEBUG (VideoThread.__init__): Config loaded.")
        except AttributeError as e:
            error_msg = f"Erreur de configuration (config.py): Attribut manquant '{e}'"
            print(f"ERREUR: {error_msg}")
            raise RuntimeError(error_msg) # Raise here to prevent thread start

        # Initialize members
        self.cnn_feature_extractor_model = None
        self.preprocess_function = None
        self.cnn_target_size = None
        self.lstm_prediction_model = None
        self.vocabulaire = None
        self.index_to_word = None
        self.cap = None
        self.mp_hands = None
        self.hands_solution = None
        self.mp_drawing = None
        self.drawing_spec_hand = Colors.MP_HAND_DRAWING_SPEC
        self.drawing_spec_connection = Colors.MP_CONNECTION_DRAWING_SPEC
        self.last_hands_detected_status = False
        print("DEBUG (VideoThread.__init__): Initialized variables.")

    # ... (load_vocabulary, load_models_and_preprocessing remain the same) ...
    def load_vocabulary(self):
        vocab = {}
        print(f"DEBUG (VideoThread.load_vocabulary): Attempting to load from {self.VOCABULARY_PATH}")
        try:
            if not os.path.exists(self.VOCABULARY_PATH):
                 raise FileNotFoundError(f"Vocabulary file not found at '{self.VOCABULARY_PATH}'")
            with open(self.VOCABULARY_PATH, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or ':' not in line:
                        if line: logging.warning(f"Line {line_num} format incorrect in '{self.VOCABULARY_PATH}': '{line}' - Skipping")
                        continue
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[0] and parts[1].isdigit():
                        vocab[parts[0].strip().lower()] = int(parts[1].strip())
                    else:
                        logging.warning(f"Line {line_num} ignored in '{self.VOCABULARY_PATH}': '{line}' - Format issue after split.")
                        print(f"DEBUG (VideoThread.load_vocabulary): Line {line_num} ignored: '{line}'")
            if not vocab:
                error_msg = f"Vocabulary loaded from '{self.VOCABULARY_PATH}' is empty or invalid."
                logging.error(error_msg)
                self.error_occurred.emit(error_msg)
                print(f"DEBUG (VideoThread.load_vocabulary): Loaded vocabulary is empty: {self.VOCABULARY_PATH}")
                return None
            logging.info(f"Vocabulary loaded successfully ({len(vocab)} words).")
            print(f"DEBUG (VideoThread.load_vocabulary): Success ({len(vocab)} words).")
            return vocab
        except FileNotFoundError as e:
            error_msg = str(e)
            logging.error(f"Error: {error_msg}")
            self.error_occurred.emit(error_msg)
            print(f"DEBUG (VideoThread.load_vocabulary): File not found: {self.VOCABULARY_PATH}")
            return None
        except Exception as e:
            error_msg = f"Error loading vocabulary from '{self.VOCABULARY_PATH}': {e}"
            logging.exception(error_msg)
            self.error_occurred.emit(error_msg)
            print(f"DEBUG (VideoThread.load_vocabulary): Error loading vocab: {e}")
            return None

    def load_models_and_preprocessing(self):
        print("DEBUG (VideoThread.load_models_and_preprocessing): Attempting to load models...")
        model_name = self.CNN_MODEL_CHOICE
        input_shape = self.CNN_INPUT_SHAPE
        self.cnn_target_size = input_shape[:2]
        logging.info(f"Loading CNN: {model_name} with input shape {input_shape}...")
        print(f"DEBUG (VideoThread.load_models_and_preprocessing): Loading CNN: {model_name}")
        try:
            if model_name == 'MobileNetV2':
                base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
                self.preprocess_function = mobilenet_preprocess
            elif model_name == 'ResNet50':
                base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
                self.preprocess_function = resnet_preprocess
            elif model_name == 'EfficientNetB0':
                base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
                self.preprocess_function = efficientnet_preprocess
            else:
                raise ValueError(f"Unsupported CNN model specified in config: {model_name}")
            output = GlobalAveragePooling2D()(base.output)
            self.cnn_feature_extractor_model = Model(inputs=base.input, outputs=output, name=f"{model_name}_FeatureExtractor")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): CNN model {model_name} structure created. Initializing...")
            # Initialize by passing a dummy input (important for some backends/TF versions)
            dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
            _ = self.cnn_feature_extractor_model(dummy_input, training=False) # Warm-up call
            logging.info(f"CNN model {model_name} loaded and initialized.")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): CNN model {model_name} loaded and initialized.")
        except Exception as e:
            error_msg = f"Error loading CNN model '{model_name}': {e}"
            logging.exception(error_msg)
            self.error_occurred.emit(error_msg) # Use emit for thread-safe UI update
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): CRITICAL ERROR loading CNN model: {e}")
            return False # Indicate failure

        logging.info(f"Loading LSTM model from: {self.MODEL_PATH}...")
        print(f"DEBUG (VideoThread.load_models_and_preprocessing): Loading LSTM model from: {self.MODEL_PATH}")
        try:
            if not os.path.exists(self.MODEL_PATH):
                 raise FileNotFoundError(f"LSTM model file not found: {self.MODEL_PATH}")
            self.lstm_prediction_model = tf.keras.models.load_model(self.MODEL_PATH)
            logging.info(f"LSTM model loaded from {self.MODEL_PATH}")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): LSTM model loaded. Checking input shape...")
            # Input shape validation
            expected_lstm_shape = self.lstm_prediction_model.input_shape
            logging.info(f"Expected LSTM input shape: {expected_lstm_shape}")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): Expected LSTM input shape: {expected_lstm_shape}")
            # Basic shape check (should be like (None, sequence_length, feature_dim))
            if not isinstance(expected_lstm_shape, (list, tuple)) or len(expected_lstm_shape) != 3:
                raise ValueError(f"LSTM model input shape has unexpected rank or type: {expected_lstm_shape}")

            model_seq_len = expected_lstm_shape[1] # Typically the sequence length (e.g., FIXED_LENGTH)
            if model_seq_len is not None and model_seq_len != self.FIXED_LENGTH:
                 logging.warning(f"LSTM Sequence Length Mismatch Warning! Model expects {model_seq_len}, config.FIXED_LENGTH is {self.FIXED_LENGTH}. Padding/truncation will occur based on config.")
                 print(f"DEBUG: WARN - LSTM sequence length mismatch: Model={model_seq_len}, Config={self.FIXED_LENGTH}")
                 # Proceed, assuming padding/truncation handles it

            model_feat_dim = expected_lstm_shape[2] # Feature dimension
            if model_feat_dim is not None and model_feat_dim != self.FEATURE_DIM:
                 # This is usually critical!
                 raise ValueError(f"CRITICAL LSTM Feature Dimension Mismatch! Model expects {model_feat_dim}, config.ACTIVE_FEATURE_DIM is {self.FEATURE_DIM}.")

            # Initialize/Warm-up LSTM
            dummy_lstm_input = tf.zeros((1, self.FIXED_LENGTH, self.FEATURE_DIM), dtype=tf.float32)
            _ = self.lstm_prediction_model(dummy_lstm_input, training=False)
            logging.info("LSTM model initialized.")
            print("DEBUG (VideoThread.load_models_and_preprocessing): LSTM model initialized.")

        except Exception as e:
            error_msg = f"Error loading or initializing LSTM model '{self.MODEL_PATH}': {e}"
            logging.exception(error_msg)
            self.error_occurred.emit(f"Erreur chargement LSTM: {e}") # Use emit
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): Error loading/initializing LSTM model: {e}")
            return False # Indicate failure

        print("DEBUG (VideoThread.load_models_and_preprocessing): Models loaded successfully.")
        return True

    # ... (extract_cnn_features_realtime remains the same) ...
    def extract_cnn_features_realtime(self, frame):
        """Tente d'utiliser cv2.resize pour potentiellement accélérer."""
        if self.cnn_feature_extractor_model is None or self.preprocess_function is None or self.cnn_target_size is None:
            logging.error("CNN model, preprocessing function, or target size not initialized.")
            print("DEBUG (VideoThread.extract_cnn_features_realtime): CNN feature extractor not initialized.")
            return None
        try:
            # 1. Redimensionnement avec OpenCV (plus rapide sur CPU?)
            # Note: cv2 utilise (width, height), tf.image.resize utilise (height, width)
            # self.cnn_target_size est (height, width), donc on l'inverse pour cv2
            target_size_cv2 = (self.cnn_target_size[1], self.cnn_target_size[0])
            # Utiliser INTER_AREA pour le rétrécissement est souvent un bon compromis vitesse/qualité
            img_resized_cv = cv2.resize(frame, target_size_cv2, interpolation=cv2.INTER_AREA)

            # 2. Conversion en Tenseur TF (après redim.)
            # Il faut s'assurer que le type est correct (souvent float32)
            img_resized_tensor = tf.convert_to_tensor(img_resized_cv, dtype=tf.float32)

            # 3. Ajout de la dimension Batch (avec TF)
            img_batch_tensor = tf.expand_dims(img_resized_tensor, axis=0)

            # 4. Prétraitement (avec TF)
            # self.preprocess_function est (par exemple) mobilenet_preprocess de Keras
            img_preprocessed_tensor = self.preprocess_function(img_batch_tensor)

            # 5. Inférence CNN (avec TF)
            features_tensor = self.cnn_feature_extractor_model(img_preprocessed_tensor, training=False)

            # 6. Conversion en NumPy SEULEMENT à la fin (pour l'ajout à la deque)
            # On prend le premier élément du batch [0] et on convertit.
            return features_tensor[0].numpy()

        except Exception as e:
            # Log less verbosely unless debugging deep issues
            logging.warning(f"Error during CV2Resize+TF CNN feature extraction: {e}", exc_info=False)
            print(f"DEBUG (VideoThread.extract_cnn_features_realtime): ERROR during extraction: {type(e).__name__}: {e}")
            # import traceback # Décommenter pour voir la trace complète si besoin
            # traceback.print_exc()
            return None

    # ... (run method remains the same - it only needs to emit prediction_ready) ...
    def run(self):
        self._running = True
        logging.info("Video processing thread started.")
        print("DEBUG: VideoThread run() started")

        # --- TensorFlow GPU Configuration ---
        print("DEBUG (VideoThread.run): Configuring TensorFlow/GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Attempt to set memory growth for all detected GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configured for memory growth: {gpus}")
                print(f"DEBUG (VideoThread.run): GPU(s) configured: {gpus}")
            except RuntimeError as e:
                # Log error but continue (TensorFlow might still work on CPU or with default GPU allocation)
                logging.error(f"Error configuring GPU memory growth: {e}")
                print(f"DEBUG (VideoThread.run): GPU config error (memory growth): {e}")
        else:
            logging.warning("No GPU detected by TensorFlow. Inference will run on CPU.")
            print("DEBUG (VideoThread.run): No GPU detected by TensorFlow.")
        print(f"DEBUG (VideoThread.run): GPU check done.")

        # --- Load Models ---
        print("DEBUG (VideoThread.run): Attempting to load models...")
        models_ok = self.load_models_and_preprocessing()
        print(f"DEBUG (VideoThread.run): Models loaded OK: {models_ok}")
        if not models_ok:
            # Emit failure signal and stop the thread
            if self._running: self.models_loaded.emit(False)
            self._running = False
            logging.error("Failed to load models. Video thread stopping.")
            print("DEBUG (VideoThread.run): Exiting run() due to model load failure")
            return # Stop execution here

        # --- Load Vocabulary ---
        print("DEBUG (VideoThread.run): Attempting to load vocabulary...")
        self.vocabulaire = self.load_vocabulary()
        print(f"DEBUG (VideoThread.run): Vocabulary loaded: {'OK' if self.vocabulaire else 'FAILED'}")
        if not self.vocabulaire:
            # Emit failure signal and stop the thread
            if self._running: self.models_loaded.emit(False) # Signal overall failure
            self._running = False
            logging.error("Failed to load vocabulary. Video thread stopping.")
            print("DEBUG (VideoThread.run): Exiting run() due to vocab load failure")
            return # Stop execution here

        # --- Create Inverse Vocabulary ---
        try:
            self.index_to_word = {i: word for word, i in self.vocabulaire.items()}
            # Check for potential issues (duplicate indices)
            if len(self.index_to_word) != len(self.vocabulaire):
                logging.warning("Potential duplicate indices found in vocabulary file.")
            logging.info(f"Inverse vocabulary created ({len(self.index_to_word)} entries).")
            print(f"DEBUG (VideoThread.run): Inverse vocabulary created ({len(self.index_to_word)} words).")
        except Exception as e:
             error_msg = f"Error creating inverse vocabulary mapping: {e}"
             logging.error(error_msg)
             if self._running: self.error_occurred.emit(error_msg) # Use emit
             if self._running: self.models_loaded.emit(False) # Signal overall failure
             self._running = False
             print(f"DEBUG (VideoThread.run): Exiting run() due to inverse vocab error")
             return # Stop execution here

        # --- Initialize Mediapipe (if available) ---
        if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            print("DEBUG (VideoThread.run): Initializing Mediapipe Hands...")
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils # Already potentially set in Colors
            try:
                # Ensure drawing specs are valid (might have failed in Colors)
                if self.drawing_spec_hand is None:
                    self.drawing_spec_hand = self.mp_drawing.DrawingSpec(color=Colors.CV_GREEN, thickness=2, circle_radius=2)
                if self.drawing_spec_connection is None:
                    self.drawing_spec_connection = self.mp_drawing.DrawingSpec(color=Colors.CV_RED, thickness=2)

                # Initialize Hands solution
                self.hands_solution = self.mp_hands.Hands(
                    static_image_mode=False, # Use video mode
                    max_num_hands=self.MAX_HANDS,
                    min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE,
                    min_tracking_confidence=self.MIN_HAND_TRACKING_CONFIDENCE)
                print(f"DEBUG (VideoThread.run): Mediapipe Hands initialized (max_hands={self.MAX_HANDS}, det_conf={self.MIN_HAND_DETECTION_CONFIDENCE:.2f}, track_conf={self.MIN_HAND_TRACKING_CONFIDENCE:.2f}).")
            except Exception as e_mp:
                print(f"ERREUR (VideoThread.run): Failed to initialize Mediapipe Hands: {e_mp}")
                if self._running: self.error_occurred.emit(f"Erreur initialisation Mediapipe: {e_mp}")
                self.hands_solution = None # Disable hand detection features
        else:
            print("WARNING (VideoThread.run): Mediapipe module or hands solution not found/loaded. Hand detection optimization disabled.")
            self.hands_solution = None
            self.drawing_spec_hand = None # Ensure these are None if mp fails
            self.drawing_spec_connection = None

        # --- Signal Success and Open Camera ---
        if self._running: self.models_loaded.emit(True) # Signal that setup (models, vocab) is complete
        print("DEBUG (VideoThread.run): Emitted models_loaded(True)")

        logging.info(f"Opening camera capture source: {self.CAPTURE_SOURCE}")
        print(f"DEBUG (VideoThread.run): Attempting to open camera source: {self.CAPTURE_SOURCE} (type: {type(self.CAPTURE_SOURCE)})")
        self.cap = None
        capture_backend = cv2.CAP_ANY # Default backend
        if sys.platform == "win32":
            capture_backend = cv2.CAP_DSHOW # Prefer DirectShow on Windows
            print("DEBUG (VideoThread.run): Using preferred cv2.CAP_DSHOW backend on Windows")

        try:
            # Determine if source is an index or string
            source_to_open = int(self.CAPTURE_SOURCE) if str(self.CAPTURE_SOURCE).isdigit() else self.CAPTURE_SOURCE
            print(f"DEBUG (VideoThread.run): Calling cv2.VideoCapture({source_to_open}, {capture_backend})")
            self.cap = cv2.VideoCapture(source_to_open, capture_backend)
            time.sleep(0.5) # Allow camera to initialize

            is_opened = self.cap.isOpened() if self.cap else False
            print(f"DEBUG (VideoThread.run): Camera is opened after initial attempt: {is_opened}")

            # Fallback for Windows if DSHOW fails
            if not is_opened and sys.platform == "win32" and capture_backend == cv2.CAP_DSHOW:
                 print("DEBUG (VideoThread.run): CAP_DSHOW failed, trying default backend (CAP_ANY)...")
                 if self.cap: self.cap.release() # Release the failed capture
                 capture_backend = cv2.CAP_ANY # Switch to default
                 self.cap = cv2.VideoCapture(source_to_open, capture_backend) # Try again
                 time.sleep(0.5)
                 is_opened = self.cap.isOpened() if self.cap else False
                 print(f"DEBUG (VideoThread.run): Camera opened with default backend: {is_opened}")

            if not is_opened:
                # If still not opened after fallbacks, raise error
                raise IOError(f"Unable to open camera source '{source_to_open}' with tested backends.")

        except Exception as e_cap:
             logging.error(f"Error opening camera capture {self.CAPTURE_SOURCE}: {e_cap}", exc_info=True)
             error_msg = f"Erreur ouverture webcam {self.CAPTURE_SOURCE}: {e_cap}"
             if self._running: self.error_occurred.emit(error_msg) # Use emit
             # Signal overall failure since camera is crucial
             if self._running: self.models_loaded.emit(False)
             self._running = False # Stop the thread
             print(f"DEBUG (VideoThread.run): Exiting run() because camera failed to open.")
             # Cleanup resources if partially initialized
             if self.cap: self.cap.release()
             if self.hands_solution: self.hands_solution.close()
             return # Stop execution

        logging.info("Webcam opened successfully.")
        print("DEBUG (VideoThread.run): Webcam opened successfully.")

        # --- Initialize Loop Variables ---
        sequence_window = deque(maxlen=self.FIXED_LENGTH)
        prediction_display_buffer = deque(maxlen=self.SMOOTHING_WINDOW_SIZE)
        frame_processing_times = deque(maxlen=30) # For FPS calculation
        frame_count = 0
        last_smoothed_word = "?" # Store the last emitted word to avoid repeats
        print("DEBUG (VideoThread.run): Real-time loop variables initialized.")

        # --- Determine Display Frame Size ---
        target_width = None
        target_height = None
        resize_needed = False
        try:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width <= 0 or frame_height <= 0:
                raise ValueError("Invalid frame dimensions from camera")
            logging.info(f"Webcam native resolution: {frame_width}x{frame_height}")
            print(f"DEBUG (VideoThread.run): Native camera resolution: {frame_width}x{frame_height}")

            # Check if resizing is needed based on config
            if self.MAX_FRAME_WIDTH and frame_width > self.MAX_FRAME_WIDTH:
                scale = self.MAX_FRAME_WIDTH / frame_width
                target_width = self.MAX_FRAME_WIDTH
                target_height = int(frame_height * scale)
                # Ensure height is even (sometimes required by codecs/display)
                target_height = target_height if target_height % 2 == 0 else target_height + 1
                resize_needed = True
                logging.info(f"Display resizing enabled: Target width {target_width}px (height ~{target_height}px)")
                print(f"DEBUG (VideoThread.run): Display will be resized to {target_width}x{target_height}")
            else:
                 target_width = frame_width # Use native size if no resize needed
                 target_height = frame_height
                 print("DEBUG (VideoThread.run): No display resizing needed based on MAX_FRAME_WIDTH.")
        except Exception as e_res:
            logging.warning(f"Could not read camera resolution: {e_res}. Using fallback display size.")
            print(f"DEBUG (VideoThread.run): Could not get camera resolution: {e_res}")
            # Fallback values if reading resolution fails
            target_width = 640
            target_height = 480
            resize_needed = True # Assume resize is needed for fallback
            print(f"DEBUG (VideoThread.run): Falling back to display size {target_width}x{target_height}")

        print("DEBUG (VideoThread.run): Entering main video loop...")
        loop_count = 0 # Counter for debugging loop iterations

        # --- Main Processing Loop ---
        while self._running:
            loop_start_time = time.time()
            loop_count += 1

            # --- Read Frame ---
            try:
                ret, frame = self.cap.read()
            except Exception as e_read:
                 logging.error(f"Exception during cap.read() (iteration {loop_count}): {e_read}", exc_info=True)
                 if self._running: self.error_occurred.emit(f"Erreur lecture webcam: {e_read}")
                 print(f"DEBUG: Breaking loop... cap.read(): {e_read}")
                 break # Exit loop on read error

            if not ret or frame is None:
                logging.error(f"Failed to read frame (iteration {loop_count}, ret={ret}, frame is None={frame is None}). Stopping thread.")
                is_still_opened = self.cap.isOpened() if self.cap else False
                error_msg = f"Impossible de lire la frame (tentative {loop_count}). Caméra ouverte: {is_still_opened}"
                if self._running: self.error_occurred.emit(error_msg)
                print(f"DEBUG: Breaking loop... cannot read frame. Camera still opened: {is_still_opened}")
                break # Exit loop if frame reading fails

            frame_count += 1

            # --- Prepare Display Frame (Resize if needed) ---
            display_frame = None
            if resize_needed and target_width and target_height:
                try:
                    display_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                except Exception as e_resize:
                    logging.warning(f"Error resizing display frame: {e_resize}")
                    display_frame = frame.copy() # Use original if resize fails
            else:
                display_frame = frame.copy() # Use copy to avoid modifying original

            # --- Hand Detection (Mediapipe) ---
            hands_detected_this_frame = False
            if self.hands_solution:
                try:
                    # Process with Mediapipe
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MP needs RGB
                    image_rgb.flags.writeable = False # Performance optimization
                    results = self.hands_solution.process(image_rgb)
                    image_rgb.flags.writeable = True # Make writeable again if needed later

                    # Draw landmarks on the *display* frame
                    if results.multi_hand_landmarks:
                        hands_detected_this_frame = True
                        for hand_landmarks in results.multi_hand_landmarks:
                            if self.mp_drawing and self.drawing_spec_hand and self.drawing_spec_connection:
                                try:
                                    self.mp_drawing.draw_landmarks(
                                        display_frame, # Draw on the potentially resized frame
                                        hand_landmarks,
                                        self.mp_hands.HAND_CONNECTIONS,
                                        self.drawing_spec_hand,
                                        self.drawing_spec_connection)
                                except Exception as e_draw:
                                    # Log drawing errors but don't stop the loop
                                    logging.warning(f"Error drawing hand landmarks: {e_draw}", exc_info=False)
                                    print(f"DEBUG Loop {loop_count}: Warning - error drawing landmarks: {e_draw}")

                except Exception as e_hand_detect:
                    logging.warning(f"Error during Mediapipe hand processing: {e_hand_detect}", exc_info=False)
                    print(f"DEBUG Loop {loop_count}: Warning - error during hand detection: {e_hand_detect}")

            # --- Emit Hand Detection Status Change ---
            if hands_detected_this_frame != self.last_hands_detected_status:
                if self._running: self.hands_detected_signal.emit(hands_detected_this_frame)
                self.last_hands_detected_status = hands_detected_this_frame
                print(f"DEBUG Loop {loop_count}: Hands detected status changed: {hands_detected_this_frame}")

            # --- Decide Whether to Run Inference ---
            should_run_inference = False
            # Condition 1: Hand detection enabled AND hands are detected, OR hand detection disabled
            hand_check_passed = (not self.hands_solution) or hands_detected_this_frame
            # Condition 2: Frame skipping interval check
            frame_interval_check_passed = (frame_count % (self.FRAMES_TO_SKIP + 1) == 0)

            if hand_check_passed and frame_interval_check_passed:
                should_run_inference = True

            # --- Run Inference if Conditions Met ---
            if should_run_inference:
                inference_start_time = time.time()
                cnn_features = self.extract_cnn_features_realtime(frame) # Use original frame for features
                processing_time_ms = (time.time() - inference_start_time) * 1000
                frame_processing_times.append(processing_time_ms) # Record time

                if cnn_features is not None:
                    sequence_window.append(cnn_features)
                    current_sequence_len = len(sequence_window)

                    if current_sequence_len > 0: # Only predict if we have features
                        padded_sequence = None
                        current_sequence_np = np.array(sequence_window, dtype=np.float32)

                        # --- Pad Sequence ---
                        if current_sequence_len < self.FIXED_LENGTH:
                            padding_size = self.FIXED_LENGTH - current_sequence_len
                            # Use numpy for padding (potentially simpler than tf.pad here)
                            padding_array = np.zeros((padding_size, self.FEATURE_DIM), dtype=np.float32)
                            padded_sequence = np.concatenate((padding_array, current_sequence_np), axis=0)
                        else: # Sequence is already full or longer (deque handles maxlen)
                            padded_sequence = current_sequence_np

                        # --- LSTM Prediction ---
                        if padded_sequence is not None and padded_sequence.shape == (self.FIXED_LENGTH, self.FEATURE_DIM):
                            # Reshape for LSTM (add batch dimension)
                            reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                            try:
                                # Perform prediction
                                prediction_probs = self.lstm_prediction_model(reshaped_sequence, training=False).numpy()[0] # Get probs for the single batch item

                                # --- Process Predictions ---
                                # Get top N predictions and confidences
                                top_n_indices = np.argsort(prediction_probs)[-self.TOP_N:][::-1] # Indices of top N probs
                                top_n_confidences = prediction_probs[top_n_indices] # Corresponding confidences
                                top_n_words = [self.index_to_word.get(idx, f"UNK_{idx}") for idx in top_n_indices] # Map indices to words

                                # Format for status bar display
                                top_n_display_list = [f"{word} ({conf:.2f})" for word, conf in zip(top_n_words, top_n_confidences)]
                                if self._running: self.top_n_ready.emit(top_n_display_list) # Emit for UI update

                                # Get the single top prediction
                                top_pred_idx = top_n_indices[0]
                                top_pred_conf = top_n_confidences[0]

                                # Add to smoothing buffer ONLY if above threshold
                                if top_pred_conf >= self.PREDICTION_THRESHOLD:
                                    prediction_display_buffer.append(top_pred_idx)
                                    # print(f"DEBUG Loop {loop_count}: Prediction '{top_n_words[0]}' ({top_pred_conf:.2f}) added to buffer.") # Optional debug
                                # else: print(f"DEBUG Loop {loop_count}: Prediction below threshold ({top_pred_conf:.2f} < {self.PREDICTION_THRESHOLD:.2f}).") # Optional debug

                            except Exception as e_pred:
                                logging.exception(f"Error during LSTM prediction: {e_pred}")
                                print(f"DEBUG: Exception during LSTM predict: {e_pred}")
                                if self._running: self.top_n_ready.emit(["Erreur Prediction LSTM"]) # Emit error status
                        else:
                            # This case indicates an issue with padding or sequence shape
                            if padded_sequence is not None: print(f"DEBUG: Incorrect sequence shape before LSTM: {padded_sequence.shape}")
                            else: print("DEBUG: Padded sequence is None.")
                            if self._running: self.top_n_ready.emit(["Erreur Shape Séquence"]) # Emit error status
                else:
                    # CNN feature extraction failed
                    print(f"DEBUG Loop {loop_count}: CNN Feature extraction returned None.")
                    if self._running: self.top_n_ready.emit(["Erreur Extraction CNN"]) # Emit error status
            # --- End of Inference Block ---

            # --- Reset Buffers if Hands Disappear (if using hand detection) ---
            elif self.hands_solution and not hands_detected_this_frame:
                # If hand detection is enabled and no hands are detected, clear buffers
                if sequence_window:
                    sequence_window.clear()
                    print(f"DEBUG Loop {loop_count}: Hands disappeared/not detected, clearing sequence window.")
                if prediction_display_buffer:
                    prediction_display_buffer.clear()
                    print(f"DEBUG Loop {loop_count}: Hands disappeared/not detected, clearing prediction buffer.")
                    # Optionally clear the UI prediction display immediately
                    if self.last_hands_detected_status: # Only if they *just* disappeared
                         if self._running: self.top_n_ready.emit([""]) # Clear status bar
                         if last_smoothed_word != "?":
                             last_smoothed_word = "?"
                             if self._running: self.prediction_ready.emit(last_smoothed_word) # Emit "?" to potentially clear UI

            # --- Smoothing and Emission ---
            current_smoothed_word = "?" # Default if buffer is empty or smoothing fails
            if prediction_display_buffer:
                try:
                    # Find the most common word index in the buffer
                    word_counts = Counter(prediction_display_buffer)
                    most_common_prediction = word_counts.most_common(1)
                    if most_common_prediction:
                        smoothed_index = most_common_prediction[0][0]
                        current_smoothed_word = self.index_to_word.get(smoothed_index, "?") # Map index back to word
                        # print(f"DEBUG Loop {loop_count}: Smoothed word determined: '{current_smoothed_word}' from buffer {list(prediction_display_buffer)}") # Optional debug
                    # else: print(f"DEBUG Loop {loop_count}: Smoothing buffer not empty, but most_common returned nothing.") # Optional debug
                except Exception as e_smooth:
                    logging.warning(f"Error during prediction smoothing: {e_smooth}")
                    print(f"DEBUG Loop {loop_count}: Exception during smoothing: {e_smooth}")

            # Emit the prediction *only if it has changed*
            if current_smoothed_word != last_smoothed_word:
                # print(f"DEBUG Loop {loop_count}: Emitting smoothed prediction: '{current_smoothed_word}' (Previous: '{last_smoothed_word}')") # Optional debug
                if self._running: self.prediction_ready.emit(current_smoothed_word)
                last_smoothed_word = current_smoothed_word
            # else: print(f"DEBUG Loop {loop_count}: Smoothed word '{current_smoothed_word}' same as last, not emitting.") # Optional debug


            # --- Draw Debug Info (FPS, Hand Status) ---
            try:
                 # Processing time FPS
                 if frame_processing_times:
                     avg_proc_time = np.mean(frame_processing_times)
                     fps_proc_approx = 1000 / avg_proc_time if avg_proc_time > 0 else 0
                     cv2.putText(display_frame, f"Proc: {avg_proc_time:.1f}ms (~{fps_proc_approx:.1f} FPS)",
                                 (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)
                 # Total loop time FPS
                 loop_time_ms = (time.time() - loop_start_time) * 1000
                 fps_loop_approx = 1000 / loop_time_ms if loop_time_ms > 0 else 0
                 cv2.putText(display_frame, f"Loop: {loop_time_ms:.1f}ms (~{fps_loop_approx:.1f} FPS)",
                             (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)

                 # Hand detection status text
                 if self.hands_solution:
                     status_text = "Mains: Oui" if hands_detected_this_frame else "Mains: Non"
                     status_color = Colors.CV_GREEN if hands_detected_this_frame else Colors.CV_RED
                     cv2.putText(display_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

            except Exception as e_display_debug:
                logging.warning(f"Error drawing debug info: {e_display_debug}", exc_info=False)
                print(f"DEBUG Loop {loop_count}: Warning - error drawing debug info: {e_display_debug}")

            # --- Emit Frame for UI ---
            try:
                 if display_frame is not None and display_frame.size > 0:
                      if self._running: self.frame_ready.emit(display_frame)
                 # else: print(f"DEBUG Loop {loop_count}: display_frame is None or empty, skipping emit.") # Optional debug
            except Exception as e_emit:
                logging.warning(f"Error emitting frame signal: {e_emit}", exc_info=False)
                print(f"DEBUG Loop {loop_count}: Warning - error emitting frame: {e_emit}")

        # --- End of Main Loop ---
        print(f"DEBUG: Exited main video loop after {loop_count} iterations.")
        logging.info("Video thread loop finished or stopped.")

        # --- Cleanup ---
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
                logging.info("Webcam released.")
                print("DEBUG: Camera released.")
            except Exception as e_rel:
                logging.error(f"Exception releasing camera: {e_rel}")
        else:
            print("DEBUG: Camera was not open or already released.")

        if self.hands_solution:
            try:
                self.hands_solution.close()
                print("DEBUG: Mediapipe Hands solution closed.")
            except Exception as e_mp_close:
                print(f"DEBUG: Error closing Mediapipe: {e_mp_close}")

        # Attempt to clear Keras session to free GPU memory (optional but good practice)
        try:
            print("DEBUG: Attempting to clear Keras session...")
            tf.keras.backend.clear_session()
            logging.info("Keras/TensorFlow session cleared.")
            print("DEBUG: Keras session cleared.")
        except Exception as e_clear:
            logging.warning(f"Error clearing Keras session: {e_clear}")
            print(f"DEBUG: Error clearing Keras session: {e_clear}")

        logging.info("Video thread finished cleanly.")
        print("DEBUG: VideoThread run() finished")

    def stop(self):
        print("DEBUG: VideoThread stop() called")
        self._running = False # Signal the loop to stop
        logging.info("Stop requested for video thread.")


# --- TTS Worker Components (Adapted from TTS_test.py) ---

# Global TTS Queue and Stop Event
tts_queue: queue.Queue = queue.Queue()
stop_worker = threading.Event()
playback_speed: float = 1.0 # Default playback speed

# Voice mapping (Dynamically created from imported Voice enum)
api_voice_keys: Dict[str, Voice] = {}
valid_voice_keys: List[str] = []
default_voice_key: str = 'fr_female' # Default voice

def initialize_voice_keys():
    """Populates voice key mappings."""
    global api_voice_keys, valid_voice_keys, default_voice_key
    try:
        # Map lowercase enum names to enum members for API voices
        api_voice_keys = {v.name.lower(): v for v in Voice}
        # All valid keys: API keys + gTTS key
        valid_voice_keys = list(api_voice_keys.keys())
        if gTTS: # Only add fr_female if gTTS is available
            valid_voice_keys.append('fr_female')
            valid_voice_keys.sort() # Keep it sorted
            default_voice_key = 'fr_female'
            print(f"DEBUG: TTS Voices Initialized. API: {len(api_voice_keys)}, gTTS: fr_female. Valid keys: {valid_voice_keys}")
        else:
            valid_voice_keys.sort()
            # Set a default from API voices if gTTS is unavailable
            if 'en_us_001' in valid_voice_keys: # Pick a common default
                 default_voice_key = 'en_us_001'
            elif valid_voice_keys: # Pick the first available API voice
                 default_voice_key = valid_voice_keys[0]
            else: # No voices available at all
                 default_voice_key = None
            print(f"DEBUG: TTS Voices Initialized (gTTS UNAVAILABLE). API: {len(api_voice_keys)}. Valid keys: {valid_voice_keys}")
            if default_voice_key is None:
                 print("ERROR: No TTS voices available (gTTS failed, API enum empty?)")

    except Exception as e:
        print(f"ERROR initializing voice keys: {e}")
        # Set safe defaults
        api_voice_keys = {}
        valid_voice_keys = []
        default_voice_key = None


def _load_tts_endpoints() -> List[Dict[str, str]]:
    """Loads API endpoints from data/config.json"""
    script_dir = os.path.dirname(__file__)
    # Path relative to mainwindow.py
    json_file_path = os.path.join(script_dir, 'data', 'config.json')
    print(f"DEBUG (_load_tts_endpoints): Trying to load endpoints from: {json_file_path}")
    if not os.path.exists(json_file_path):
        print(f"ERROR: TTS config file not found at {json_file_path}")
        # Create a dummy file? Or just return empty? Let's return empty.
        # You might want to create a default dummy file here for the user
        print(f"INFO: Please create '{json_file_path}' with your API endpoints.")
        # Example dummy creation:
        dummy_config = [
             {"url": "https://tiktok-tts.weilnet.workers.dev/api/generation", "response": "data"},
             {"url": "YOUR_SECOND_ENDPOINT_HERE", "response": "audio_base64"}
        ]
        try:
             data_dir = os.path.dirname(json_file_path)
             if not os.path.exists(data_dir): os.makedirs(data_dir)
             with open(json_file_path, 'w') as f:
                 json_dump(dummy_config, f, indent=4)
             print(f"INFO: Created a default dummy TTS config at '{json_file_path}'. EDIT IT!")
             return dummy_config # Return the dummy for now
        except Exception as e_create:
             print(f"ERROR: Could not create dummy TTS config: {e_create}")
             return [] # Return empty if creation failed

    try:
        with open(json_file_path, 'r') as file:
            data = json_load(file)
            print(f"DEBUG (_load_tts_endpoints): Successfully loaded {len(data)} endpoints.")
            return data
    except Exception as e:
        print(f"ERROR loading TTS endpoints from {json_file_path}: {e}")
        return []

def _split_tts_text(text: str) -> List[str]:
    """Splits text for the TTS API (copied from TTS_test.py)."""
    merged_chunks: List[str] = []
    # Split primarily by sentence-ending punctuation, preserving punctuation
    separated_chunks: List[str] = re.findall(r'[^.!?]+[.!?]*', text) # Simple split
    character_limit: int = 200 # Reduced limit for TTS APIs, often stricter than 300

    processed_chunks: List[str] = []
    for chunk in separated_chunks:
        chunk = chunk.strip()
        if not chunk: continue
        # Encode to check byte length, as some APIs might have byte limits
        if len(chunk.encode("utf-8")) > character_limit:
             # If a sentence is too long, split by spaces (less ideal)
             # Find words/spaces, aiming for max char limit per sub-chunk
             sub_chunks = []
             current_sub = ""
             for part in re.findall(r'\S+\s*', chunk): # Get word + trailing space
                 if len((current_sub + part).encode("utf-8")) <= character_limit:
                     current_sub += part
                 else:
                     if current_sub: sub_chunks.append(current_sub.strip())
                     # Start new sub_chunk, handle case where single part is too long
                     if len(part.encode("utf-8")) <= character_limit:
                         current_sub = part
                     else: # Single word/part exceeds limit, add it as is (API might fail)
                         print(f"WARNING: Single text part exceeds char limit ({character_limit}): '{part[:30]}...'")
                         if current_sub: sub_chunks.append(current_sub.strip()) # Add previous first
                         sub_chunks.append(part.strip()) # Add the long part
                         current_sub = "" # Reset

             if current_sub: sub_chunks.append(current_sub.strip())
             processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(chunk)

    # Combine processed chunks back, respecting the limit (final check)
    current_chunk: str = ""
    for separated_chunk in processed_chunks:
        if len((current_chunk + " " + separated_chunk).encode("utf-8")) <= character_limit and current_chunk:
             current_chunk += " " + separated_chunk # Add with space if not first part
        elif len(separated_chunk.encode("utf-8")) <= character_limit and not current_chunk:
             current_chunk = separated_chunk # Start new chunk
        else:
            if current_chunk: # Add the completed chunk
                 merged_chunks.append(current_chunk)
            # Start new chunk with the current separated_chunk (if it fits)
            if len(separated_chunk.encode("utf-8")) <= character_limit:
                current_chunk = separated_chunk
            else:
                 # Should not happen if previous split worked, but as fallback:
                 print(f"WARNING: Chunk still too long after final merge: '{separated_chunk[:30]}...'")
                 if current_chunk: merged_chunks.append(current_chunk) # Add previous chunk
                 merged_chunks.append(separated_chunk) # Add the oversized chunk
                 current_chunk = "" # Reset

    if current_chunk: # Add the last chunk
        merged_chunks.append(current_chunk)

    final_chunks = [chunk for chunk in merged_chunks if chunk and not chunk.isspace()]
    print(f"DEBUG (_split_tts_text): Split '{text[:30]}...' into {len(final_chunks)} chunks.")
    return final_chunks


def _fetch_audio_bytes_from_api_chunk(
    endpoint: Dict[str, str],
    text_chunk: str,
    voice_id: str
) -> Optional[str]:
    """Fetches a single audio chunk (base64 encoded string) from the API."""
    # This function is adapted from TTS_test.py's version
    try:
        url = endpoint.get("url")
        response_key = endpoint.get("response")
        if not url or not response_key:
            print("  [API TTS Worker] Error: Invalid endpoint dictionary structure.")
            return None

        print(f"  [API TTS Worker] Sending chunk to {url} (Voice: {voice_id}): {text_chunk[:30]}...")
        # Use a reasonable timeout
        response = requests.post(url, json={"text": text_chunk, "voice": voice_id}, timeout=20) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_response = response.json()

        if isinstance(json_response, dict) and response_key in json_response:
             audio_data_b64 = json_response[response_key]
             if isinstance(audio_data_b64, str) and audio_data_b64:
                  print(f"  [API TTS Worker] Received valid base64 chunk data.")
                  return audio_data_b64
             else:
                  print(f"  [API TTS Worker] Error: Key '{response_key}' found but data is empty or not a string: {audio_data_b64}")
                  return None
        else:
             print(f"  [API TTS Worker] Error: Key '{response_key}' not found or response is not a dict: {json_response}")
             return None

    except requests.exceptions.Timeout:
        print(f"  [API TTS Worker] Error: Request timed out for chunk: {text_chunk[:30]}...")
        return None
    except requests.exceptions.RequestException as e:
        # Handles connection errors, HTTP errors, etc.
        print(f"  [API TTS Worker] Error fetching audio chunk: {e}")
        return None
    except Exception as e:
        # Catch other potential errors (JSON decoding, etc.)
        print(f"  [API TTS Worker] An unexpected error occurred during fetch: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return None

def _save_audio_file(output_file_path: str, audio_bytes: bytes) -> bool:
    """Saves audio bytes to a file, returns True on success."""
    try:
        # Ensure directory exists if path includes folders
        # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "wb") as file:
            file.write(audio_bytes)
        print(f"  [TTS Worker Util] Audio data successfully written to {output_file_path}")
        return True
    except IOError as e:
        print(f"  [TTS Worker Util] Error writing audio file {output_file_path}: {e}")
        return False
    except Exception as e:
        print(f"  [TTS Worker Util] Unexpected error saving file {output_file_path}: {e}")
        return False

def generate_api_audio(text: str, output_file_path: str, voice: Voice) -> bool:
    """Generates audio using the API (logic adapted from tts_library and TTS_test.py)."""
    # This function orchestrates the API call process based on the worker pattern
    print(f"[API TTS Worker] Generating audio for voice {voice.name} ({voice.value}): {text[:50]}...")

    # 1. Validate basic args (already done by speak, but double check)
    if not text or not isinstance(voice, Voice):
        print(f"[API TTS Worker] Error: Invalid arguments - Text empty or voice not Voice enum.")
        return False

    # 2. Load Endpoints
    endpoint_data = _load_tts_endpoints()
    if not endpoint_data:
        print("[API TTS Worker] Error: No API endpoints loaded from data/config.json.")
        return False

    # 3. Split Text
    text_chunks: List[str] = _split_tts_text(text)
    if not text_chunks:
        print("[API TTS Worker] Error: Text resulted in empty chunks after splitting.")
        return False

    # 4. Iterate through endpoints
    for endpoint in endpoint_data:
        print(f"[API TTS Worker] Trying endpoint: {endpoint.get('url', 'N/A')}")
        num_chunks = len(text_chunks)
        # --- Use Threads to fetch chunks in parallel ---
        # List to store results (base64 strings or None on failure)
        audio_chunks_b64: List[Optional[str]] = [None] * num_chunks
        threads: List[threading.Thread] = []
        # Use a dictionary to store results safely from threads
        results_dict: Dict[int, Optional[str]] = {}

        # Target function for each thread
        def thread_target(index: int, chunk_text: str, endpoint_info: dict, voice_id_str: str):
            results_dict[index] = _fetch_audio_bytes_from_api_chunk(endpoint_info, chunk_text, voice_id_str)

        # Create and start threads
        for i, chunk in enumerate(text_chunks):
            thread = threading.Thread(target=thread_target, args=(i, chunk, endpoint, voice.value), name=f"TTSChunk-{i}")
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        # --- End Threads ---

        # Collect results from the dictionary into the list
        for i in range(num_chunks):
            audio_chunks_b64[i] = results_dict.get(i) # Get result or None if index wasn't set

        # 5. Check if all chunks were successful for this endpoint
        if all(chunk is not None for chunk in audio_chunks_b64):
            print("[API TTS Worker] All chunks received successfully for this endpoint. Concatenating and decoding...")
            try:
                # Concatenate base64 strings
                full_audio_b64 = "".join(audio_chunks_b64) # Already checked for None
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(full_audio_b64)
                # Save the final audio file
                if _save_audio_file(output_file_path, audio_bytes):
                    print(f"[API TTS Worker] Audio successfully generated and saved to {output_file_path}")
                    return True # Success! Exit the endpoint loop.
                else:
                    # Save failed, but endpoint worked. Still counts as endpoint failure for retry.
                    print(f"[API TTS Worker] Endpoint worked, but failed to save file {output_file_path}. Trying next endpoint.")
                    # No need to break, loop will continue to next endpoint

            except base64.binascii.Error as e:
                print(f"[API TTS Worker] Error decoding base64 string: {e}. Trying next endpoint.")
                # Continue to the next endpoint
            except Exception as e:
                print(f"[API TTS Worker] Unexpected error during saving/decoding: {e}. Trying next endpoint.")
                # Continue to the next endpoint
        else:
             # Calculate missing chunks for debugging
             missing_indices = [i for i, ch in enumerate(audio_chunks_b64) if ch is None]
             print(f"[API TTS Worker] Failed to fetch all audio chunks for endpoint {endpoint.get('url', 'N/A')}. Missing chunks at indices: {missing_indices}. Trying next endpoint.")
             # Loop continues to the next endpoint automatically

    # 6. If loop finishes without success
    print("[API TTS Worker] Error: Failed to generate audio using ALL available API endpoints.")
    return False

def generate_female_audio(text: str, output_file_path: str) -> bool:
    """Generates French female voice audio using gTTS."""
    if not gTTS:
        print("[FR Female TTS Worker] Error: gTTS library not available.")
        return False

    print(f"[FR Female TTS Worker - gTTS] Generating audio for: {text[:50]}...")
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='fr', slow=False)
        # Save directly to the output file path
        tts.save(output_file_path)
        print(f"[FR Female TTS Worker - gTTS] Audio successfully saved to {output_file_path}")
        return True
    except Exception as e:
        print(f"[FR Female TTS Worker - gTTS] Error generating or saving audio: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return False

def speed_change(sound: AudioSegment, speed: float = 1.0) -> Optional[AudioSegment]:
    """Changes the speed of an AudioSegment using frame rate adjustment."""
    if not AudioSegment: return None # Check if pydub is available
    if speed == 1.0:
        return sound
    if speed <= 0:
        print(f"  [Playback Worker] Error: Invalid speed ({speed}). Must be positive.")
        return None # Return None for invalid speed

    print(f"  [Playback Worker] Applying speed factor: {speed:.2f}")
    try:
        # Simple (but pitch-affecting) speed change by altering frame rate
        new_frame_rate = int(sound.frame_rate * speed)
        if new_frame_rate <= 0:
            print(f"  [Playback Worker] Warning: Calculated frame rate ({new_frame_rate}) is invalid. Using original rate.")
            return sound # Return original sound if rate is invalid

        # Create a new segment with the modified frame rate
        changed_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
        return changed_sound

    except Exception as e:
        print(f"  [Playback Worker] Error applying speed change: {e}")
        return sound # Return original sound on error


def play_audio_file(file_path: str):
    """Plays an audio file using pydub, applying global speed."""
    global playback_speed # Use the global speed setting

    if not PYDUB_AVAILABLE or not AudioSegment or not pydub_play:
        print("[Playback Worker] Pydub/Playback function not available. Skipping playback.")
        return

    if not os.path.exists(file_path):
         print(f"[Playback Worker] Error: File not found for playback: {file_path}")
         return

    audio_to_play = None # Initialize for error handling clarity

    try:
        print(f"[Playback Worker] Loading audio file: {file_path}...")
        # Explicitly specify format if possible, otherwise let pydub detect
        file_extension = os.path.splitext(file_path)[1].lower().strip('.')
        if not file_extension: file_extension = "mp3" # Default guess

        # Load the audio file
        audio = AudioSegment.from_file(file_path, format=file_extension)
        print(f"  [Playback Worker] Audio loaded: Duration={len(audio)/1000.0:.2f}s, Rate={audio.frame_rate}Hz")

        # Apply speed change (if speed is not 1.0)
        audio_at_speed = speed_change(audio, playback_speed)
        if audio_at_speed is None: # Handle error from speed_change
             print("[Playback Worker] Speed change failed. Playing at normal speed.")
             audio_at_speed = audio # Fallback to original

        # --- Resampling (Optional but often needed for compatibility) ---
        # Some backends prefer standard rates like 44100 Hz, especially after frame rate manipulation
        target_frame_rate = 44100 # A common standard rate
        current_rate = audio_at_speed.frame_rate

        # Resample if the rate is different AND speed wasn't 1.0 (as speed_change already altered it)
        # Or force resample if the current rate is unusual (e.g., very low/high)
        force_resample = not (8000 <= current_rate <= 96000) # Example range of 'reasonable' rates
        if force_resample or (playback_speed != 1.0 and int(current_rate) != target_frame_rate):
            print(f"  [Playback Worker] Resampling from {current_rate} Hz to {target_frame_rate} Hz for compatibility...")
            try:
                # Resample the (potentially speed-changed) audio
                audio_to_play = audio_at_speed.set_frame_rate(target_frame_rate)
            except Exception as e_resample:
                print(f"  [Playback Worker] Error during resampling: {e_resample}. Attempting playback with original rate.")
                audio_to_play = audio_at_speed # Fallback to the non-resampled version
        else:
            audio_to_play = audio_at_speed # No resampling needed

        # --- Play ---
        print(f"[Playback Worker] Playing at {playback_speed:.2f}x speed (Final Sample Rate: {audio_to_play.frame_rate} Hz)...")
        pydub_play(audio_to_play)
        print(f"[Playback Worker] Finished playing {os.path.basename(file_path)}.")

    except FileNotFoundError:
        # This check is technically redundant due to the check at the start, but good practice
        print(f"[Playback Worker] Error: File disappeared before playback? {file_path}")
    except Exception as e:
        # Catch potential pydub errors (backend issues, format issues, ffmpeg path)
        print(f"[Playback Worker] Error playing sound file {file_path}: {e}")
        # Provide hints for common issues
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower() or "[winerror 2]" in str(e).lower():
             print("[Playback Worker] HINT: Ensure 'ffmpeg'/'ffprobe' executables are installed and in your system PATH.")
        elif "permission denied" in str(e).lower():
             print("[Playback Worker] HINT: Check file/folder permissions, especially for temporary directories if used by the playback backend.")
        elif "backend" in str(e).lower():
            print("[Playback Worker] HINT: Audio playback backend issue. Try installing different backends like 'simpleaudio' or 'sounddevice'.")
        else:
             print("[Playback Worker] HINT: Check audio file integrity and format compatibility.")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug trace

def tts_worker():
    """Worker thread that processes TTS requests from the queue."""
    print("[TTS Worker] Worker thread started.")
    initialize_voice_keys() # Initialize mappings when worker starts

    while not stop_worker.is_set():
        try:
            # Wait for a request from the queue (with timeout to allow checking stop_worker)
            # Queue item format: (text_to_speak: str, voice_key_string: str)
            text, voice_key = tts_queue.get(timeout=1.0)
            print(f"\n[TTS Worker] Processing request: VoiceKey='{voice_key}', Text='{text[:50]}...'")

            success = False
            temp_file_path = None # Store path for cleanup

            try:
                # Create a temporary file for the audio output
                # Using delete=False allows us to control deletion after playback
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                    temp_file_path = tmp_audio_file.name
                    # tmp_audio_file.close() # Close handle immediately on some systems? Usually handled by 'with'.
                print(f"[TTS Worker] Using temporary file: {temp_file_path}")

                # --- Generation Logic ---
                if voice_key == 'fr_female':
                    # Use gTTS for French Female voice
                    success = generate_female_audio(text, temp_file_path)
                elif voice_key in api_voice_keys:
                    # Use API for other voices
                    target_voice_enum = api_voice_keys[voice_key] # Find the corresponding Voice enum member
                    success = generate_api_audio(text, temp_file_path, target_voice_enum)
                else:
                    # Should not happen if 'speak' validates, but handle anyway
                    print(f"[TTS Worker] Error: Unknown voice key '{voice_key}' received in worker.")
                    success = False
                # --- End Generation Logic ---

                # --- Playback ---
                if success and temp_file_path and os.path.exists(temp_file_path):
                    print(f"[TTS Worker] Generation successful. Attempting playback...")
                    play_audio_file(temp_file_path)
                elif not success:
                    print(f"[TTS Worker] Audio generation failed for request.")
                # If file doesn't exist here (e.g., save failed), playback won't happen

            except Exception as e_process:
                 print(f"[TTS Worker] Error during TTS processing/playback for '{text[:30]}...': {e_process}")
                 # import traceback; traceback.print_exc() # Uncomment for detailed debugging
            finally:
                # --- Cleanup Temporary File ---
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"[TTS Worker] Cleaned up temporary file: {temp_file_path}")
                    except OSError as e:
                        # Log error but don't crash worker
                        print(f"[TTS Worker] Error removing temporary file {temp_file_path}: {e}")

            # Signal that this task is done
            tts_queue.task_done()

        except queue.Empty:
            # Queue was empty, just continue loop to check stop_worker again
            continue
        except Exception as e:
            # Catch unexpected errors in the worker loop itself
            print(f"[TTS Worker] Unexpected error in worker loop: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for serious errors
            # Attempt to mark task done if possible, might fail if error was before get()
            try:
                tts_queue.task_done()
            except ValueError: # task_done called too many times
                 pass
            time.sleep(1) # Avoid busy-looping on continuous errors

    print("[TTS Worker] Worker thread finished.")

def speak(text: str, voice_key: str):
    """Adds a text-to-speech request to the queue."""
    # Validate inputs before queuing
    if not text or text.isspace():
        print("[TTS Frontend] Error: Text cannot be empty.")
        return
    if not voice_key:
         print("[TTS Frontend] Error: Voice key cannot be empty.")
         return

    voice_key_lower = voice_key.lower() # Use lowercase for comparisons

    # Check if the voice key is valid
    if voice_key_lower not in valid_voice_keys:
        print(f"[TTS Frontend] Error: Invalid voice_key '{voice_key}'. Valid keys are: {valid_voice_keys}")
        # Optional: Fallback to default? Or just reject? Rejecting is safer.
        return

    # Special checks for disabled features
    if voice_key_lower == 'fr_female' and not gTTS:
        print(f"[TTS Frontend] Error: Voice '{voice_key}' selected, but gTTS is not available.")
        return
    if not PYDUB_AVAILABLE:
        print(f"[TTS Frontend] Info: Pydub not available, queuing TTS request for '{voice_key}' but playback will be skipped.")
        # Allow queuing for generation, but worker won't play.

    print(f"[TTS Frontend] Queuing request: VoiceKey={voice_key_lower}, Text='{text[:50]}...'")
    # Put the validated request into the queue
    tts_queue.put((text, voice_key_lower))

# --- End TTS Worker Components ---


# --- Ui_ParametersWindow Class (Add Voice Selection) ---
class Ui_ParametersWindow(QWidget):
    color_changed = Signal(QColor)
    bg_color_changed = Signal(QColor)
    voice_changed = Signal(str) # Signal to emit the selected voice key (string)

    def setupUi(self, ParametersWindow):
        ParametersWindow.setObjectName(u"ParametersWindow")
        ParametersWindow.resize(400, 350) # Increased height for voice selector
        ParametersWindow.setWindowTitle("Paramètres d'Affichage et Voix")

        self.main_layout = QVBoxLayout(ParametersWindow)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.label = QLabel("Paramètres d'Affichage et Voix", ParametersWindow)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(u"font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        self.main_layout.addWidget(self.label)

        # --- Color Settings Group ---
        color_group_box = QFrame(ParametersWindow)
        color_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        color_layout = QGridLayout(color_group_box)
        color_layout.setVerticalSpacing(10)
        color_layout.setHorizontalSpacing(10)

        self.text_color_label = QLabel("Couleur du texte:", color_group_box)
        self.text_color_btn = QPushButton("Choisir", color_group_box)
        self.text_color_btn.clicked.connect(self.choose_text_color)
        self.text_color_preview = QLabel(color_group_box)
        self.text_color_preview.setFixedSize(25, 25)
        self.text_color_preview.setStyleSheet("background-color: white; border: 1px solid black; border-radius: 3px;")
        self.text_color_preview.setToolTip("Aperçu couleur texte")

        self.bg_color_label = QLabel("Couleur de fond:", color_group_box)
        self.bg_color_btn = QPushButton("Choisir", color_group_box)
        self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.bg_color_preview = QLabel(color_group_box)
        self.bg_color_preview.setFixedSize(25, 25)
        self.bg_color_preview.setStyleSheet("background-color: rgb(10, 32, 77); border: 1px solid black; border-radius: 3px;")
        self.bg_color_preview.setToolTip("Aperçu couleur fond")

        color_layout.addWidget(self.text_color_label, 0, 0)
        color_layout.addWidget(self.text_color_preview, 0, 1)
        color_layout.addWidget(self.text_color_btn, 0, 2)
        color_layout.addWidget(self.bg_color_label, 1, 0)
        color_layout.addWidget(self.bg_color_preview, 1, 1)
        color_layout.addWidget(self.bg_color_btn, 1, 2)
        color_layout.setColumnStretch(0, 1)
        color_layout.setColumnStretch(2, 0)
        self.main_layout.addWidget(color_group_box)

        # --- Voice Selection Group ---
        voice_group_box = QFrame(ParametersWindow)
        voice_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        voice_layout = QHBoxLayout(voice_group_box) # Horizontal layout for label and combobox

        self.voice_label = QLabel("Voix TTS:", voice_group_box)
        self.voice_combobox = QComboBox(voice_group_box)
        self.voice_combobox.setObjectName(u"voice_combobox")
        self.voice_combobox.setToolTip("Choisir la voix pour la synthèse vocale")
        # Populate ComboBox externally after UI setup
        self.voice_combobox.currentIndexChanged.connect(self.on_voice_selection_changed)

        voice_layout.addWidget(self.voice_label)
        voice_layout.addWidget(self.voice_combobox, 1) # Combobox takes available space
        self.main_layout.addWidget(voice_group_box)


        self.main_layout.addStretch(1) # Pushes buttons to bottom

        # --- Bottom Buttons ---
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addStretch(1)
        self.default_btn = QPushButton("Par défaut", ParametersWindow)
        self.default_btn.setToolTip("Réinitialiser les couleurs et la voix par défaut")
        self.default_btn.clicked.connect(self.reset_defaults)
        self.buttons_layout.addWidget(self.default_btn)

        self.close_btn = QPushButton("Fermer", ParametersWindow)
        self.close_btn.clicked.connect(ParametersWindow.close)
        self.buttons_layout.addWidget(self.close_btn)
        self.main_layout.addLayout(self.buttons_layout)

        ParametersWindow.setLayout(self.main_layout)

    def populate_voices(self, available_keys: List[str], current_key: Optional[str]):
        """Populates the voice combobox."""
        self.voice_combobox.blockSignals(True) # Prevent signal emission during population
        self.voice_combobox.clear()
        current_index = -1
        # Sort keys for consistent display
        sorted_keys = sorted(available_keys) if available_keys else []
        for i, key in enumerate(sorted_keys):
            # Use a slightly more user-friendly name if possible
            display_name = key.replace('_', ' ').title()
            # Special case for gTTS voice
            if key == 'fr_female': display_name = "Français Femme (gTTS)"
            elif key.startswith('en_us_'): display_name = f"Anglais US {key.split('_')[-1]} ({'F' if 'female' in key else 'M'})"
            elif key.startswith('en_uk_'): display_name = f"Anglais UK {key.split('_')[-1]} ({'F' if 'female' in key else 'M'})"
            # Add more cases as needed for better display names...

            self.voice_combobox.addItem(display_name, userData=key) # Store original key in userData
            if key == current_key:
                current_index = i

        if current_index != -1:
            self.voice_combobox.setCurrentIndex(current_index)
        elif self.voice_combobox.count() > 0:
             # If current key wasn't found but list isn't empty, select the first one
             self.voice_combobox.setCurrentIndex(0)
             # Emit the change for the newly selected default
             self.on_voice_selection_changed(0)


        self.voice_combobox.blockSignals(False) # Re-enable signals

    @Slot()
    def choose_text_color(self):
        parent = self.parentWidget() if self.parentWidget() else self
        current = self.text_color_preview.palette().window().color()
        color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur du texte")
        if color.isValid():
            self.text_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;")
            self.color_changed.emit(color)
            print(f"DEBUG (Ui_ParametersWindow): Text color chosen: {color.name()}")

    @Slot()
    def choose_bg_color(self):
        parent = self.parentWidget() if self.parentWidget() else self
        current = self.bg_color_preview.palette().window().color()
        color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur de fond")
        if color.isValid():
            self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;")
            self.bg_color_changed.emit(color)
            print(f"DEBUG (Ui_ParametersWindow): Background color chosen: {color.name()}")

    @Slot()
    def reset_defaults(self):
        global default_voice_key # Access the global default
        default_text = QColor("white")
        default_bg = QColor(10, 32, 77)
        default_voice = default_voice_key if default_voice_key else "" # Use global default

        # Reset colors
        self.text_color_preview.setStyleSheet(f"background-color: {default_text.name()}; border: 1px solid black; border-radius: 3px;")
        self.bg_color_preview.setStyleSheet(f"background-color: {default_bg.name()}; border: 1px solid black; border-radius: 3px;")
        self.color_changed.emit(default_text)
        self.bg_color_changed.emit(default_bg)

        # Reset voice selection
        found_index = self.voice_combobox.findData(default_voice)
        if found_index != -1:
            self.voice_combobox.setCurrentIndex(found_index)
            # Manually emit signal if index changed, as setCurrentIndex might not if already there
            if self.voice_combobox.currentData() == default_voice:
                 self.voice_changed.emit(default_voice)
        elif self.voice_combobox.count() > 0:
             self.voice_combobox.setCurrentIndex(0) # Select first if default not found
             if self.voice_combobox.currentData(): # Emit the first item's key
                  self.voice_changed.emit(self.voice_combobox.currentData())


        print(f"DEBUG (Ui_ParametersWindow): Settings reset to defaults (Voice: {default_voice}).")

    @Slot(int)
    def on_voice_selection_changed(self, index):
        """Emits the selected voice key when the ComboBox changes."""
        if index >= 0:
            selected_key = self.voice_combobox.itemData(index)
            if selected_key:
                print(f"DEBUG (Ui_ParametersWindow): Voice selection changed to index {index}, key: '{selected_key}'")
                self.voice_changed.emit(selected_key) # Emit the original key


# --- Ui_MainWindow Class (Structure verticale - No changes needed here) ---
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(800, 750) # Ajuster la taille par défaut si besoin
        MainWindow.setWindowTitle("Traduction LSF en Temps Réel (Avec Audio TTS)") # Updated title

        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        MainWindow.setSizePolicy(sizePolicy)

        self.default_bg_color = QColor(10, 32, 77)
        self.default_text_color = QColor("white")

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        # Layout principal gridLayout_3 (une seule colonne)
        self.gridLayout_3 = QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(10, 10, 10, 10) # Ajouter marges
        self.gridLayout_3.setVerticalSpacing(15) # Espacement vertical

        # --- Rangée 0: Barre d'outils ---
        self.setup_top_toolbar(self.centralwidget)
        # Ajout à la grille: Ligne 0, Colonne 0
        self.gridLayout_3.addLayout(self.gridLayout_top_toolbar, 0, 0)

        # --- Rangée 1: Vue caméra ---
        self.setup_camera_view(self.centralwidget)
        # Ajout à la grille: Ligne 1, Colonne 0
        self.gridLayout_3.addWidget(self.frame_camera, 1, 0) # Ajoute le frame directement

        # --- Rangée 2: Zone de texte ---
        self.setup_text_area(self.centralwidget)
         # Ajout à la grille: Ligne 2, Colonne 0
        self.gridLayout_3.addWidget(self.frame_text, 2, 0) # Ajoute le frame directement

        # --- Rangée 3: Contrôles (Export - Keep disabled for now) ---
        self.setup_export_controls(self.centralwidget)
        # Ajout à la grille: Ligne 3, Colonne 0
        self.gridLayout_3.addLayout(self.horizontalLayout_export, 3, 0, Qt.AlignmentFlag.AlignCenter) # Centre le bouton horizontalement

        # --- Définir les stretchs pour la grille principale ---
        self.gridLayout_3.setRowStretch(0, 0) # Toolbar: hauteur fixe
        self.gridLayout_3.setRowStretch(1, 2) # Camera: prend plus de place verticale (ratio 2)
        self.gridLayout_3.setRowStretch(2, 1) # Text Area: prend moins de place (ratio 1)
        self.gridLayout_3.setRowStretch(3, 0) # Export: hauteur fixe
        self.gridLayout_3.setColumnStretch(0, 1) # Une seule colonne qui s'étire

        # --- Configuration finale ---
        MainWindow.setCentralWidget(self.centralwidget)
        self.setup_menu_statusbar(MainWindow)
        self.retranslateUi(MainWindow)
        # QMetaObject.connectSlotsByName sera appelé par MainWindow.__init__

    # --- setup_ methods remain largely the same ---
    def setup_top_toolbar(self, parent):
        self.gridLayout_top_toolbar = QGridLayout()
        self.gridLayout_top_toolbar.setObjectName(u"gridLayout_TopToolbar")
        # Settings Button
        self.boutonparametre = QPushButton(parent)
        self.boutonparametre.setObjectName(u"boutonparametre")
        self.boutonparametre.setFixedSize(QSize(50, 50))
        self.boutonparametre.setToolTip("Ouvrir les paramètres d'affichage et voix") # Updated tooltip
        self.boutonparametre.setText("⚙️")
        self.boutonparametre.setFont(QFont("Segoe UI Emoji", 16)) # Ensure font supports emoji
        self.boutonparametre.setStyleSheet("QPushButton {border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white;} QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); }")
        self.gridLayout_top_toolbar.addWidget(self.boutonparametre, 0, 0, Qt.AlignmentFlag.AlignLeft)

        # Logo/Title
        self.logo = QLabel(parent)
        self.logo.setObjectName(u"logo")
        self.logo.setText("Traduction LSF")
        self.logo.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: transparent;")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_top_toolbar.addWidget(self.logo, 0, 1) # Span across middle columns potentially

        # Spacer to push logo left and keep settings button on the left edge
        spacerItem = QSpacerItem(50, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self.gridLayout_top_toolbar.addItem(spacerItem, 0, 2) # Add spacer to the right

        self.gridLayout_top_toolbar.setColumnStretch(1, 1) # Let the logo column take the space

    def setup_camera_view(self, parent):
        self.frame_camera = QFrame(parent)
        self.frame_camera.setObjectName(u"frame_camera")
        sizePolicyCamFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_camera.setSizePolicy(sizePolicyCamFrame)
        self.frame_camera.setMinimumHeight(300) # Minimum height for camera view
        self.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }") # Style with object name

        gridLayout_camera_inner = QGridLayout(self.frame_camera)
        gridLayout_camera_inner.setObjectName(u"gridLayout_camera_inner")
        gridLayout_camera_inner.setContentsMargins(1, 1, 1, 1) # Minimal margin inside frame

        self.camera_view = QLabel(self.frame_camera)
        self.camera_view.setObjectName(u"camera_view")
        sizePolicyCamLabel = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow label to scale
        self.camera_view.setSizePolicy(sizePolicyCamLabel)
        self.camera_view.setStyleSheet(u"QLabel#camera_view { background-color: transparent; border: none; color: grey; font-size: 16pt; }") # Style with object name
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setScaledContents(False) # We handle scaling manually in update_frame
        gridLayout_camera_inner.addWidget(self.camera_view, 0, 0, 1, 1)

    def setup_text_area(self, parent):
        self.frame_text = QFrame(parent)
        self.frame_text.setObjectName(u"frame_text")
        sizePolicyTextFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Expand horizontally, limited vertically
        self.frame_text.setSizePolicy(sizePolicyTextFrame)
        self.frame_text.setMinimumHeight(100) # Min height
        self.frame_text.setMaximumHeight(250) # Max height
        self.frame_text.setStyleSheet("QFrame#frame_text { background-color: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px; padding: 5px; }")

        layout_inside_frame = QVBoxLayout(self.frame_text)
        layout_inside_frame.setContentsMargins(5, 5, 5, 5)
        layout_inside_frame.setSpacing(5)

        self.label_predictions = QLabel("Prédictions:", self.frame_text)
        self.label_predictions.setObjectName(u"label_predictions")
        font_pred = QFont(); font_pred.setPointSize(11); font_pred.setBold(True)
        self.label_predictions.setFont(font_pred)
        self.label_predictions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.label_predictions.setStyleSheet("background-color: transparent; color: white;") # Set initial color
        layout_inside_frame.addWidget(self.label_predictions)

        self.textEdit = QTextEdit(self.frame_text)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setStyleSheet(u"QTextEdit { font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: white; }") # Set initial color
        self.textEdit.setReadOnly(True)
        layout_inside_frame.addWidget(self.textEdit, 1) # TextEdit takes remaining space (stretch factor 1)

    def setup_export_controls(self, parent):
        # Keep export disabled as it's not implemented
        self.horizontalLayout_export = QHBoxLayout()
        self.horizontalLayout_export.setObjectName(u"horizontalLayout_export")
        self.verticalLayout_export = QVBoxLayout()
        self.verticalLayout_export.setObjectName(u"verticalLayout_export")

        self.exportation = QPushButton(parent)
        self.exportation.setObjectName(u"exportation")
        self.exportation.setFixedSize(QSize(50, 50))
        self.exportation.setText("💾")
        self.exportation.setFont(QFont("Segoe UI Emoji", 16))
        self.exportation.setToolTip("Exporter le texte (Non implémenté)")
        self.exportation.setEnabled(False) # Keep disabled
        self.exportation.setStyleSheet("QPushButton { border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.3); } QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); } QPushButton:disabled { background-color: rgba(128, 128, 128, 0.2); color: gray; border-color: rgba(128, 128, 128, 0.4); }")
        self.verticalLayout_export.addWidget(self.exportation)
        self.horizontalLayout_export.addLayout(self.verticalLayout_export)

    def setup_menu_statusbar(self, MainWindow):
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setStyleSheet("QStatusBar { color: #DDDDDD; padding-left: 5px; background-color: transparent; }") # Initial style
        MainWindow.setStatusBar(self.statusbar)

        self.menubar = QMenuBar(MainWindow) # Keep menubar even if empty for standard look
        self.menubar.setObjectName(u"menubar")
        MainWindow.setMenuBar(self.menubar)

    def retranslateUi(self, MainWindow):
        # Use standard _translate method
        _translate = QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", u"Traduction LSF en Temps Réel (Avec Audio TTS)", None)) # Already set in setupUi

        # Tooltips
        if hasattr(self, 'boutonparametre'): self.boutonparametre.setToolTip(_translate("MainWindow", u"Ouvrir les paramètres d'affichage et voix", None))
        if hasattr(self, 'exportation'): self.exportation.setToolTip(_translate("MainWindow", u"Exporter le texte (Non implémenté)", None))

        # Initial Text/Placeholders
        if hasattr(self, 'logo') and self.logo.text() == "": self.logo.setText(_translate("MainWindow", u"Traduction LSF", None))
        if hasattr(self, 'camera_view'): self.camera_view.setText(_translate("MainWindow", u"Initialisation...", None))
        if hasattr(self, 'textEdit'): self.textEdit.setPlaceholderText(_translate("MainWindow", u"Les mots prédits apparaîtront ici...", None))

        # Labels
        if hasattr(self, 'label_predictions'): self.label_predictions.setText(_translate("MainWindow", u"Prédictions :", None))
        # if hasattr(self, 'label_export'): self.label_export.setText(_translate("MainWindow", u"Exporter", None)) # Export label removed


# --- MainWindow Class (Application Logic - Updated) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        print("DEBUG (MainWindow.__init__): UI setup complete.")

        # --- TTS Initialization ---
        self.tts_thread = None
        self.selected_voice_key = default_voice_key # Initialize with default
        print(f"DEBUG (MainWindow.__init__): Initial selected TTS voice key: {self.selected_voice_key}")
        self.start_tts_worker() # Start the worker thread

        # --- Parameters Window ---
        self.parameters_window = None
        self.ui_parameters = None # To hold the UI object for the parameters window
        self.ui.boutonparametre.clicked.connect(self.open_parameters)
        print("DEBUG (MainWindow.__init__): Parameters button connected.")

        # --- Styling ---
        self.current_text_color = self.ui.default_text_color
        self.current_bg_color = self.ui.default_bg_color
        self.apply_initial_styles()
        print("DEBUG (MainWindow.__init__): Initial styles applied.")

        # --- Camera Placeholder ---
        self.placeholder_timer = QTimer(self)
        self.placeholder_timer.timeout.connect(self.show_placeholder_frame)
        self.placeholder_frame_counter = 0
        self.placeholder_active = True # Start with placeholder active
        self.ui.camera_view.setText("Initialisation...")
        self.ui.camera_view.setAlignment(Qt.AlignCenter)
        print("DEBUG (MainWindow.__init__): Placeholder timer setup.")

        # --- Video Thread ---
        self.video_thread = VideoThread(self)
        print("DEBUG (MainWindow.__init__): VideoThread instance created.")
        # Connect signals from VideoThread to slots in MainWindow
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.prediction_ready.connect(self.update_prediction) # Connect prediction signal
        self.video_thread.top_n_ready.connect(self.update_top_n_status)
        self.video_thread.error_occurred.connect(self.handle_error)
        self.video_thread.models_loaded.connect(self.on_models_loaded)
        self.video_thread.finished.connect(self.on_thread_finished)
        # self.video_thread.hands_detected_signal.connect(self.update_hand_detection_status) # Keep if needed for UI feedback
        print("DEBUG (MainWindow.__init__): VideoThread signals connected.")

        # --- Start Process ---
        self.ui.statusbar.showMessage("Initialisation: Chargement des modèles et de la caméra...")
        self.placeholder_timer.start(50) # Start placeholder animation timer (50ms interval)
        print("DEBUG (MainWindow.__init__): Placeholder timer started.")
        print("DEBUG (MainWindow.__init__): Starting VideoThread...")
        self.video_thread.start() # Start the video processing thread

    def start_tts_worker(self):
        """Starts the TTS worker thread if not already running."""
        global stop_worker
        if self.tts_thread and self.tts_thread.isRunning():
            print("DEBUG (MainWindow.start_tts_worker): TTS worker already running.")
            return

        # Ensure stop event is clear before starting
        stop_worker.clear()
        self.tts_thread = threading.Thread(target=tts_worker, name="TTSWorker", daemon=True)
        self.tts_thread.start()
        print("DEBUG (MainWindow.start_tts_worker): TTS worker thread started.")

    def stop_tts_worker(self):
        """Signals the TTS worker thread to stop and waits for it."""
        global stop_worker
        if self.tts_thread and self.tts_thread.is_alive():
            print("DEBUG (MainWindow.stop_tts_worker): Signaling TTS worker to stop...")
            stop_worker.set()
            # Wait briefly for the worker to finish tasks and exit cleanly
            self.tts_thread.join(timeout=2.0) # Wait max 2 seconds
            if self.tts_thread.is_alive():
                print("WARNING (MainWindow.stop_tts_worker): TTS worker did not exit cleanly after timeout.")
            else:
                print("DEBUG (MainWindow.stop_tts_worker): TTS worker stopped.")
            self.tts_thread = None
        else:
            print("DEBUG (MainWindow.stop_tts_worker): TTS worker not running or already stopped.")


    def apply_initial_styles(self):
         print("DEBUG (MainWindow.apply_initial_styles): Applying initial styles.")
         bg_color_name = self.current_bg_color.name()
         # Apply to centralwidget for overall background
         style = f"QWidget#centralwidget {{ background-color: {bg_color_name}; }}"
         self.ui.centralwidget.setStyleSheet(style)
         # Apply text colors to relevant widgets
         self.update_text_colors(self.current_text_color)

    @Slot()
    def show_placeholder_frame(self):
        if not self.placeholder_active: return
        try:
            label = self.ui.camera_view
            if not label or not label.isVisible() or label.width() <= 0 or label.height() <= 0:
                # print("DEBUG Placeholder: Label not ready") # Too verbose
                return

            w = label.width()
            h = label.height()
            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.GlobalColor.black) # Use global color

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Simple pulsing dot animation
            center_x = w / 2.0
            center_y = h / 2.0
            max_radius = min(w, h) / 7.0 # Relative size
            radius_variation = max_radius / 2.5
            # Smooth pulse using sine wave
            pulse = (1 + np.sin(self.placeholder_frame_counter * 0.15)) / 2.0 # 0.0 to 1.0
            current_radius = max_radius - (radius_variation * pulse)

            if current_radius > 0:
                painter.setBrush(QColor(50, 50, 50)) # Dark grey dot
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(center_x, center_y), current_radius, current_radius)

            # Text indicating status
            font = QFont("Arial", 14)
            painter.setFont(font)
            painter.setPen(QColor(180, 180, 180)) # Light grey text

            text = "En attente de la caméra..."
            # Check thread status for more informative message
            if self.video_thread and self.video_thread.isFinished():
                 current_status = self.ui.statusbar.currentMessage().upper() # Check status bar message
                 if "ERREUR" in current_status or "ÉCHEC" in current_status or "FAILED" in current_status:
                     text = "Échec initialisation / Erreur"
                 else:
                     text = "Caméra déconnectée"

            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.end()

            label.setPixmap(pixmap)
            self.placeholder_frame_counter += 1

        except Exception as e:
             print(f"ERROR (MainWindow.show_placeholder_frame): {e}")
             # Stop the timer on error to prevent spamming logs
             if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
             self.placeholder_active = False # Ensure it stops trying
             # Display error message directly on label if possible
             if label: label.setText(f"Erreur Placeholder:\n{e}")


    @Slot(np.ndarray)
    def update_frame(self, cv_img):
        # Stop placeholder once the first valid frame arrives
        if self.placeholder_active:
            print("DEBUG (MainWindow.update_frame): First real frame received, stopping placeholder.")
            if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
            self.placeholder_active = False
            self.ui.camera_view.clear() # Clear any placeholder text/pixmap
            self.ui.camera_view.setText("") # Ensure text is empty

        if cv_img is None or cv_img.size == 0:
            print("DEBUG (MainWindow.update_frame): Received empty/invalid frame, skipping.")
            return

        try:
            # Convert OpenCV image (BGR) to QPixmap
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

            if qt_image.isNull():
                print("ERROR DEBUG (MainWindow.update_frame): QImage creation failed!")
                self.ui.camera_view.setText("Erreur: QImage Null")
                return

            qt_pixmap = QPixmap.fromImage(qt_image)
            if qt_pixmap.isNull():
                print("ERROR DEBUG (MainWindow.update_frame): QPixmap conversion failed!")
                self.ui.camera_view.setText("Erreur: QPixmap Null")
                return

            # Scale pixmap to fit the label while keeping aspect ratio
            label = self.ui.camera_view
            label_size = label.size() # Get current size of the label widget

            # Check if label size is valid before scaling
            if label_size.isValid() and label_size.width() > 10 and label_size.height() > 10:
                # Scale the pixmap
                scaled_pixmap = qt_pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                if scaled_pixmap.isNull():
                    print("ERROR DEBUG (MainWindow.update_frame): Scaled QPixmap is Null! Using original.")
                    label.setPixmap(qt_pixmap) # Fallback to original pixmap
                else:
                    label.setPixmap(scaled_pixmap) # Set the scaled pixmap
            else:
                # If label size is invalid (e.g., during init), just set the original pixmap
                label.setPixmap(qt_pixmap)

        except cv2.error as e_cv:
             # Handle potential OpenCV errors during processing (though unlikely here)
             logging.error(f"OpenCV error in update_frame: {e_cv}", exc_info=True)
             print(f"DEBUG (MainWindow.update_frame): OpenCV error: {e_cv}")
             self.ui.camera_view.setText(f"Erreur OpenCV:\n{e_cv}")
             # Optionally restart placeholder if error occurs
             if not self.placeholder_active:
                 self.placeholder_active = True
                 self.placeholder_timer.start(50)
        except Exception as e:
            # Catch any other unexpected errors during Qt conversion/display
            logging.error(f"Error updating frame: {e}", exc_info=True)
            print(f"DEBUG (MainWindow.update_frame): Exception: {e}")
            self.ui.camera_view.setText(f"Erreur affichage:\n{str(e)}")
            # Optionally restart placeholder
            if not self.placeholder_active:
                 self.placeholder_active = True
                 self.placeholder_timer.start(50)

    @Slot(str)
    def update_prediction(self, text):
        """Handles the predicted word: updates text area and triggers TTS."""
        # Only process non-empty and non-placeholder predictions
        if text and text != "?":
            # --- Update Text Area ---
            current_content = self.ui.textEdit.toPlainText()
            words = current_content.split()
            # Add the new word only if it's different from the last word
            if not words or text.lower() != words[-1].lower():
                 max_words_history = 50 # Limit the history in the text box
                 new_words = words + [text]
                 if len(new_words) > max_words_history:
                     new_words = new_words[-max_words_history:] # Keep only the last N words
                 self.ui.textEdit.setPlainText(" ".join(new_words))
                 self.ui.textEdit.moveCursor(QTextCursor.MoveOperation.End) # Scroll to end

                 # --- Trigger TTS ---
                 if self.selected_voice_key: # Check if a voice is selected
                     print(f"DEBUG (MainWindow.update_prediction): Triggering TTS for '{text}' with voice '{self.selected_voice_key}'")
                     speak(text, self.selected_voice_key) # Call the function to queue TTS
                 else:
                     print("DEBUG (MainWindow.update_prediction): No TTS voice selected, skipping speech.")

    @Slot(list)
    def update_top_n_status(self, top_n_list):
        # Filter out empty strings or strings with only whitespace
        filtered_list = [item for item in top_n_list if item and item.strip()]

        if filtered_list:
            # Join the top N predictions/confidences for status bar display
            # Use the TOP_N value from the video thread's config
            display_limit = getattr(self.video_thread, 'ui_TOP_N', 3) # Default to 3 if not found
            status_text = " | ".join(filtered_list[:display_limit])
            self.ui.statusbar.showMessage(status_text)
        else:
            # If list is empty, show a default message, unless an error is already shown
            current_message = self.ui.statusbar.currentMessage().upper()
            if not current_message or ("ERREUR" not in current_message and "ÉCHEC" not in current_message):
                 # Show "Ready" briefly if no predictions and no persistent error
                 self.ui.statusbar.showMessage("Prêt.", 3000) # Show for 3 seconds

    # @Slot(bool) # Keep this slot if you want visual feedback for hand detection
    # def update_hand_detection_status(self, detected):
    #     # Example: Change border color of camera view
    #     if detected:
    #         self.ui.frame_camera.setStyleSheet("QFrame#frame_camera { border: 2px solid lime; border-radius: 5px; background-color: black; }")
    #     else:
    #         self.ui.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }")
    #     pass # Implement visual feedback if desired

    @Slot(str)
    def handle_error(self, message):
        # Log the error
        logging.error(f"Error reported from video thread: {message}")
        print(f"DEBUG (MainWindow.handle_error): Received error signal: {message}")

        # Keywords indicating a critical error that likely stops functionality
        critical_keywords = [
            "impossible d'ouvrir", "webcam", "fichier modèle", "model file", "vocabulaire",
            "shape lstm", "feature dimension", "erreur chargement cnn", "erreur chargement lstm",
            "erreur ouverture webcam", "erreur lecture webcam", "mediapipe init", "config gpu",
            "critical", "manquant", "missing", "not found", "empty", "failed to initialize",
            "attributeerror:", "indexerror:", "valueerror:", "runtimeerror:", # Include common exception types
            "camera failed", "model load failure", "vocab load failure"
        ]
        # Check if the message contains any critical keyword (case-insensitive)
        is_critical = any(keyword in message.lower() for keyword in critical_keywords)

        if is_critical:
             print(f"DEBUG (MainWindow.handle_error): Critical error identified: {message}")
             # Display prominently in status bar (permanently)
             self.ui.statusbar.showMessage(f"ERREUR CRITIQUE: {message}", 0) # 0 timeout = permanent
             # Display in camera view as well
             self.ui.camera_view.setText(f"ERREUR CRITIQUE:\n{message}\nVérifiez la console/logs.")
             self.ui.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center the error text

             # Ensure placeholder reactivates if it wasn't already
             if not self.placeholder_active:
                 print("DEBUG (MainWindow.handle_error): Reactivating placeholder due to critical error...")
                 self.placeholder_active = True
                 if not self.placeholder_timer.isActive():
                     self.placeholder_timer.start(50)

             # Show a popup message box for critical errors
             try:
                 # Run message box in the main thread using QTimer.singleShot
                 QTimer.singleShot(0, lambda: QMessageBox.critical(self,
                     "Erreur Critique",
                     f"Une erreur critique est survenue :\n\n{message}\n\n"
                     "La traduction ou la capture vidéo risque de ne pas fonctionner.\n"
                     "Veuillez vérifier la configuration (config.py), les fichiers modèle/vocabulaire, "
                     "la connexion de la caméra et les logs détaillés en console."))
             except Exception as e_msgbox:
                  print(f"ERROR: Could not display critical error message box: {e_msgbox}")

        else:
            # For non-critical errors, show temporarily in status bar
            self.ui.statusbar.showMessage(f"Erreur: {message}", 10000) # Show for 10 seconds

    @Slot(bool)
    def on_models_loaded(self, success):
        print(f"DEBUG (MainWindow.on_models_loaded): Received models_loaded signal with success={success}")
        if success:
            # Models loaded, video thread will proceed to open camera
            self.ui.statusbar.showMessage("Modèles chargés. Démarrage de la capture webcam...", 5000)
        else:
            # Models failed to load, thread should have stopped or will stop soon
            final_fail_msg = "ÉCHEC INITIALISATION: Modèles/Vocab/Webcam. Vérifiez logs."
            self.ui.statusbar.showMessage(final_fail_msg, 0) # Permanent error message
            # Update camera view placeholder text
            self.ui.camera_view.setText(f"ÉCHEC INITIALISATION:\nVérifiez la console/logs.")
            self.ui.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            print(f"DEBUG (MainWindow.on_models_loaded): Initialization failed (models_loaded=False received)")
            # Ensure placeholder is active
            if not self.placeholder_active:
                 self.placeholder_active = True
                 if not self.placeholder_timer.isActive(): self.placeholder_timer.start(50)


    @Slot()
    def on_thread_finished(self):
        """Called when the VideoThread naturally finishes or stops."""
        logging.info("Video processing thread has finished execution.")
        print("DEBUG (MainWindow.on_thread_finished): Video thread finished signal received.")

        # Update status bar unless a critical error is already displayed
        current_status = self.ui.statusbar.currentMessage().upper()
        if "ERREUR" not in current_status and "ÉCHEC" not in current_status:
            self.ui.statusbar.showMessage("Connexion caméra terminée.", 5000) # Show briefly

        # Ensure placeholder is active and shows appropriate message
        if not self.placeholder_active:
            print("DEBUG (MainWindow.on_thread_finished): Reactivating placeholder.")
            self.placeholder_active = True
            if not self.placeholder_timer.isActive(): self.placeholder_timer.start(50) # Restart animation timer

        # Update placeholder text based on why the thread might have finished
        if "ERREUR" in current_status or "ÉCHEC" in current_status:
            self.ui.camera_view.setText("Caméra déconnectée / Erreur") # Keep error state visible
        else:
             self.ui.camera_view.setText("Caméra déconnectée")
        self.ui.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)

    @Slot()
    def open_parameters(self):
        print("DEBUG (MainWindow.open_parameters): Opening parameters window...")
        if self.parameters_window is None:
            # Create the window and its UI
            self.parameters_window = QWidget(self, Qt.WindowType.Window) # Make it a top-level window
            self.parameters_window.setWindowModality(Qt.WindowModality.NonModal) # Allow interaction with main window

            self.ui_parameters = Ui_ParametersWindow() # Instantiate the UI class
            self.ui_parameters.setupUi(self.parameters_window) # Setup the UI on the window widget

            if self.ui_parameters:
                # Set initial values from MainWindow state
                self.ui_parameters.text_color_preview.setStyleSheet(f"background-color: {self.current_text_color.name()}; border: 1px solid black; border-radius: 3px;")
                self.ui_parameters.bg_color_preview.setStyleSheet(f"background-color: {self.current_bg_color.name()}; border: 1px solid black; border-radius: 3px;")

                # Populate voice combobox
                self.ui_parameters.populate_voices(valid_voice_keys, self.selected_voice_key)

                # Connect signals from parameters UI to slots in MainWindow
                self.ui_parameters.color_changed.connect(self.update_text_colors)
                self.ui_parameters.bg_color_changed.connect(self.update_bg_color)
                self.ui_parameters.voice_changed.connect(self.update_selected_voice) # Connect voice change signal

                print("DEBUG (MainWindow.open_parameters): Parameters window created and signals connected.")
            else:
                print("ERROR: Failed to setup parameters window UI.")
                # Clean up if UI setup failed
                self.parameters_window.deleteLater() # Schedule deletion
                self.parameters_window = None
                self.ui_parameters = None
                return # Abort opening
        else:
            # If window exists, just ensure it's visible and updated
            print("DEBUG (MainWindow.open_parameters): Parameters window already exists, updating values.")
            # Update previews and voice selection in case they changed programmatically
            if self.ui_parameters:
                 self.ui_parameters.text_color_preview.setStyleSheet(f"background-color: {self.current_text_color.name()}; border: 1px solid black; border-radius: 3px;")
                 self.ui_parameters.bg_color_preview.setStyleSheet(f"background-color: {self.current_bg_color.name()}; border: 1px solid black; border-radius: 3px;")
                 # Find and set current voice index
                 found_index = self.ui_parameters.voice_combobox.findData(self.selected_voice_key)
                 if found_index != -1:
                     self.ui_parameters.voice_combobox.setCurrentIndex(found_index)

        # Show, raise (bring to front), and activate the window
        self.parameters_window.show()
        self.parameters_window.raise_()
        self.parameters_window.activateWindow()

    @Slot(QColor)
    def update_text_colors(self, color):
        if not color.isValid(): return
        self.current_text_color = color
        color_name = color.name()
        print(f"DEBUG (MainWindow.update_text_colors): Applying text color: {color_name}")

        # Define styles using the new color
        text_style = f"color: {color_name}; background-color: transparent;"
        bold_text_style = f"color: {color_name}; background-color: transparent; font-weight: bold;"

        # Apply styles to relevant UI elements
        # Use hasattr for safety in case UI elements change
        if hasattr(self.ui, 'label_export'): # Though export might be removed/disabled
            self.ui.label_export.setStyleSheet(bold_text_style)
        if hasattr(self.ui, 'label_predictions'):
            self.ui.label_predictions.setStyleSheet(bold_text_style)
        if hasattr(self.ui, 'logo'):
            # Combine font size with color style
            self.ui.logo.setStyleSheet(f"font-size: 24px; {bold_text_style}")
        if hasattr(self.ui, 'statusbar'):
            # Use a slightly lighter color for status bar for contrast maybe
            status_color = color.lighter(110).name() # Example: slightly lighter
            self.ui.statusbar.setStyleSheet(f"QStatusBar {{ color: {status_color}; padding-left: 5px; background-color: transparent; }}")
        if hasattr(self.ui, 'textEdit'):
            # Combine font settings with color
            self.ui.textEdit.setStyleSheet(f"QTextEdit {{ font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: {color_name}; }}")

    @Slot(QColor)
    def update_bg_color(self, color):
        if not color.isValid(): return
        self.current_bg_color = color
        color_name = color.name()
        print(f"DEBUG (MainWindow.update_bg_color): Applying background color: {color_name}")

        # Apply background color ONLY to the central widget
        style = f"QWidget#centralwidget {{ background-color: {color_name}; }}"
        self.ui.centralwidget.setStyleSheet(style)

        # Important: Re-apply text colors AFTER changing background to ensure
        # elements with transparent backgrounds show the correct text color.
        self.update_text_colors(self.current_text_color)

    @Slot(str)
    def update_selected_voice(self, voice_key):
        """Stores the newly selected voice key."""
        if voice_key in valid_voice_keys:
            self.selected_voice_key = voice_key
            print(f"DEBUG (MainWindow.update_selected_voice): Selected TTS voice updated to: '{self.selected_voice_key}'")
        else:
             print(f"WARNING (MainWindow.update_selected_voice): Attempted to set invalid voice key: '{voice_key}'")


    def closeEvent(self, event):
        """Handles the main window closing sequence."""
        logging.info("Main window close requested.")
        print("DEBUG (MainWindow.closeEvent): Close event triggered.")

        # Stop timers
        if self.placeholder_timer.isActive():
            print("DEBUG: Stopping placeholder timer.")
            self.placeholder_timer.stop()

        # Close parameters window if it's open
        if self.parameters_window and self.parameters_window.isVisible():
            print("DEBUG: Closing parameters window.")
            self.parameters_window.close()

        # Stop the video thread
        if self.video_thread and self.video_thread.isRunning():
            logging.info("Stopping video thread...")
            print("DEBUG: Signaling video thread to stop...")
            self.video_thread.stop() # Signal the thread to stop
            print("DEBUG: Waiting for video thread...")
            # Wait for the thread to finish (max 3 seconds)
            if not self.video_thread.wait(3000):
                logging.warning("Video thread did not stop gracefully within timeout.")
                print("DEBUG: Video thread timed out.")
                # Consider termination if necessary, but wait() is preferred
            else:
                logging.info("Video thread stopped.")
                print("DEBUG: Video thread stopped.")
        else:
            logging.info("Video thread not running or already stopped.")
            print("DEBUG: Video thread not running.")

        # Stop the TTS worker thread
        self.stop_tts_worker()

        print("DEBUG (MainWindow.closeEvent): Accepting close event.")
        event.accept() # Allow the window to close

# --- Application Entry Point ---
if __name__ == "__main__":
     print("DEBUG: Application starting execution from mainwindow.py")

     # --- Pre-flight Checks ---
     # Check for data directory and config.json (used by TTS worker)
     script_dir = os.path.dirname(os.path.abspath(__file__))
     data_dir = os.path.join(script_dir, 'data')
     tts_config_file = os.path.join(data_dir, 'config.json')
     if not os.path.exists(data_dir):
         try:
             os.makedirs(data_dir)
             print(f"INFO: Created data directory: {data_dir}")
         except OSError as e:
             print(f"ERROR: Could not create data directory {data_dir}: {e}")
             # Proceed, but TTS API will likely fail later
     if not os.path.exists(tts_config_file):
          print(f"WARNING: TTS config file '{tts_config_file}' not found.")
          _load_tts_endpoints() # Attempt to create a default one

     # Check if pydub dependencies are likely met
     if not PYDUB_AVAILABLE:
          print("\n" + "="*40)
          print("WARNING: PyDub (for audio playback) is not available.")
          print("         TTS generation might work, but audio will not play.")
          print("         Install pydub: pip install pydub")
          print("         AND ensure FFmpeg is installed and in your system PATH.")
          print("="*40 + "\n")

     app = QApplication(sys.argv)
     # Set application style (optional, 'Fusion' often looks good cross-platform)
     # app.setStyle('Fusion')

     window = MainWindow()
     window.showMaximized() # Or window.show() for a normal window
     print("DEBUG: Entering Qt application event loop...")
     exit_code = app.exec()
     print(f"DEBUG: Application event loop finished with exit code {exit_code}.")
     sys.exit(exit_code)
