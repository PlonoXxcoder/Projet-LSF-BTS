# -*- coding: utf-8 -*-
# mainwindow.py

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
                             QWidget, QMessageBox, QButtonGroup)
print("DEBUG: Imported QtWidgets")
# NOTE: QtMultimedia and QtTextToSpeech imports removed

# --- Imports pour le traitement vidéo ---
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
    print("ERREUR: Le module 'mediapipe' n'est pas installé.")
    print("Veuillez l'installer avec : pip install mediapipe")
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

# NOTE: Custom TTS import removed

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
print("DEBUG: Logging configured.")

# --- Colors Class ---
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
class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    prediction_ready = Signal(str)
    top_n_ready = Signal(list)
    models_loaded = Signal(bool)
    error_occurred = Signal(str)
    hands_detected_signal = Signal(bool)

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
            self.ui_TOP_N = config.CAPTURE_TOP_N
            print("DEBUG (VideoThread.__init__): Config loaded.")
        except AttributeError as e:
            error_msg = f"Erreur de configuration (config.py): Attribut manquant '{e}'"
            print(f"ERREUR: {error_msg}")
            raise RuntimeError(error_msg)

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
            dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
            _ = self.cnn_feature_extractor_model(dummy_input, training=False)
            logging.info(f"CNN model {model_name} loaded and initialized.")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): CNN model {model_name} loaded and initialized.")
        except Exception as e:
            error_msg = f"Error loading CNN model '{model_name}': {e}"
            logging.exception(error_msg)
            self.error_occurred.emit(error_msg)
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): CRITICAL ERROR loading CNN model: {e}")
            return False

        logging.info(f"Loading LSTM model from: {self.MODEL_PATH}...")
        print(f"DEBUG (VideoThread.load_models_and_preprocessing): Loading LSTM model from: {self.MODEL_PATH}")
        try:
            if not os.path.exists(self.MODEL_PATH):
                 raise FileNotFoundError(f"LSTM model file not found: {self.MODEL_PATH}")
            self.lstm_prediction_model = tf.keras.models.load_model(self.MODEL_PATH)
            logging.info(f"LSTM model loaded from {self.MODEL_PATH}")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): LSTM model loaded. Checking input shape...")
            expected_lstm_shape = self.lstm_prediction_model.input_shape
            logging.info(f"Expected LSTM input shape: {expected_lstm_shape}")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): Expected LSTM input shape: {expected_lstm_shape}")
            if len(expected_lstm_shape) != 3: raise ValueError(f"LSTM model input shape has unexpected rank: {len(expected_lstm_shape)}")
            model_seq_len = expected_lstm_shape[1]
            if model_seq_len is not None and model_seq_len != self.FIXED_LENGTH:
                 logging.warning(f"LSTM Sequence Length Mismatch Warning! Model expects {model_seq_len}, config.FIXED_LENGTH is {self.FIXED_LENGTH}. Padding/truncation will occur.")
            model_feat_dim = expected_lstm_shape[2]
            if model_feat_dim is not None and model_feat_dim != self.FEATURE_DIM:
                 raise ValueError(f"CRITICAL LSTM Feature Dimension Mismatch! Model expects {model_feat_dim}, config.ACTIVE_FEATURE_DIM is {self.FEATURE_DIM}.")
            dummy_lstm_input = tf.zeros((1, self.FIXED_LENGTH, self.FEATURE_DIM), dtype=tf.float32)
            _ = self.lstm_prediction_model(dummy_lstm_input, training=False)
            logging.info("LSTM model initialized.")
            print("DEBUG (VideoThread.load_models_and_preprocessing): LSTM model initialized.")
        except Exception as e:
            error_msg = f"Error loading or initializing LSTM model '{self.MODEL_PATH}': {e}"
            logging.exception(error_msg)
            self.error_occurred.emit(f"Erreur chargement LSTM: {e}")
            print(f"DEBUG (VideoThread.load_models_and_preprocessing): Error loading/initializing LSTM model: {e}")
            return False
        print("DEBUG (VideoThread.load_models_and_preprocessing): Models loaded successfully.")
        return True

    # ================================================================ #
    # === FUNCTION MODIFIÉE : extract_cnn_features_realtime (cv2.resize) === #
    # ================================================================ #
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
            logging.warning(f"Error during CV2Resize+TF CNN feature extraction: {e}", exc_info=False)
            print(f"DEBUG (VideoThread.extract_cnn_features_realtime): ERROR during extraction: {type(e).__name__}: {e}")
            # import traceback # Décommenter pour voir la trace complète si besoin
            # traceback.print_exc()
            return None
    # ================================================================ #
    # === FIN DE LA MODIFICATION === #
    # ================================================================ #

    def run(self):
        self._running = True
        logging.info("Video processing thread started.")
        print("DEBUG: VideoThread run() started")
        print("DEBUG (VideoThread.run): Configuring TensorFlow/GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configured for memory growth: {gpus}")
                print(f"DEBUG (VideoThread.run): GPU(s) configured: {gpus}")
            except RuntimeError as e:
                logging.error(f"Error configuring GPU memory growth: {e}")
                print(f"DEBUG (VideoThread.run): GPU config error (memory growth): {e}")
        else:
            logging.warning("No GPU detected by TensorFlow. Inference will run on CPU.")
            print("DEBUG (VideoThread.run): No GPU detected by TensorFlow.")
        print(f"DEBUG (VideoThread.run): GPU check done.")

        print("DEBUG (VideoThread.run): Attempting to load models...")
        models_ok = self.load_models_and_preprocessing()
        print(f"DEBUG (VideoThread.run): Models loaded OK: {models_ok}")
        if not models_ok:
            self.models_loaded.emit(False); self._running = False
            logging.error("Failed to load models. Video thread stopping.")
            print("DEBUG (VideoThread.run): Exiting run() due to model load failure"); return

        print("DEBUG (VideoThread.run): Attempting to load vocabulary...")
        self.vocabulaire = self.load_vocabulary()
        print(f"DEBUG (VideoThread.run): Vocabulary loaded: {'OK' if self.vocabulaire else 'FAILED'}")
        if not self.vocabulaire:
            self.models_loaded.emit(False); self._running = False
            logging.error("Failed to load vocabulary. Video thread stopping.")
            print("DEBUG (VideoThread.run): Exiting run() due to vocab load failure"); return

        try:
            self.index_to_word = {i: word for word, i in self.vocabulaire.items()}
            if len(self.index_to_word) != len(self.vocabulaire): logging.warning("Potential duplicate indices found in vocabulary file.")
            logging.info(f"Inverse vocabulary created ({len(self.index_to_word)} entries).")
            print(f"DEBUG (VideoThread.run): Inverse vocabulary created ({len(self.index_to_word)} words).")
        except Exception as e:
             error_msg = f"Error creating inverse vocabulary mapping: {e}"
             logging.error(error_msg); self.error_occurred.emit(error_msg)
             self.models_loaded.emit(False); self._running = False
             print(f"DEBUG (VideoThread.run): Exiting run() due to inverse vocab error"); return

        if mp and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            print("DEBUG (VideoThread.run): Initializing Mediapipe Hands...")
            self.mp_hands = mp.solutions.hands; self.mp_drawing = mp.solutions.drawing_utils
            try:
                if self.drawing_spec_hand is None: self.drawing_spec_hand = self.mp_drawing.DrawingSpec(color=Colors.CV_GREEN, thickness=2, circle_radius=2)
                if self.drawing_spec_connection is None: self.drawing_spec_connection = self.mp_drawing.DrawingSpec(color=Colors.CV_RED, thickness=2)
                self.hands_solution = self.mp_hands.Hands(static_image_mode=False, max_num_hands=self.MAX_HANDS, min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_HAND_TRACKING_CONFIDENCE)
                print(f"DEBUG (VideoThread.run): Mediapipe Hands initialized (max_hands={self.MAX_HANDS}, det_conf={self.MIN_HAND_DETECTION_CONFIDENCE:.2f}, track_conf={self.MIN_HAND_TRACKING_CONFIDENCE:.2f}).")
            except Exception as e_mp:
                print(f"ERREUR (VideoThread.run): Failed to initialize Mediapipe Hands: {e_mp}")
                self.error_occurred.emit(f"Erreur initialisation Mediapipe: {e_mp}")
                self.hands_solution = None
        else:
            print("WARNING (VideoThread.run): Mediapipe module or hands solution not found/loaded. Hand detection optimization disabled.")
            self.hands_solution = None; self.drawing_spec_hand = None; self.drawing_spec_connection = None

        self.models_loaded.emit(True)
        print("DEBUG (VideoThread.run): Emitted models_loaded(True)")

        logging.info(f"Opening camera capture source: {self.CAPTURE_SOURCE}")
        print(f"DEBUG (VideoThread.run): Attempting to open camera source: {self.CAPTURE_SOURCE} (type: {type(self.CAPTURE_SOURCE)})")
        self.cap = None; capture_backend = cv2.CAP_ANY
        if sys.platform == "win32": capture_backend = cv2.CAP_DSHOW; print("DEBUG (VideoThread.run): Using preferred cv2.CAP_DSHOW backend on Windows")
        try:
            source_to_open = int(self.CAPTURE_SOURCE) if str(self.CAPTURE_SOURCE).isdigit() else self.CAPTURE_SOURCE
            print(f"DEBUG (VideoThread.run): Calling cv2.VideoCapture({source_to_open}, {capture_backend})")
            self.cap = cv2.VideoCapture(source_to_open, capture_backend); time.sleep(0.5)
            is_opened = self.cap.isOpened() if self.cap else False
            print(f"DEBUG (VideoThread.run): Camera is opened after initial attempt: {is_opened}")
            if not is_opened and sys.platform == "win32" and capture_backend == cv2.CAP_DSHOW:
                 print("DEBUG (VideoThread.run): CAP_DSHOW failed, trying default backend (CAP_ANY)...")
                 if self.cap: self.cap.release()
                 capture_backend = cv2.CAP_ANY; self.cap = cv2.VideoCapture(source_to_open, capture_backend); time.sleep(0.5)
                 is_opened = self.cap.isOpened() if self.cap else False
                 print(f"DEBUG (VideoThread.run): Camera opened with default backend: {is_opened}")
            if not is_opened: raise IOError(f"Unable to open camera source '{source_to_open}' with tested backends.")
        except Exception as e_cap:
             logging.error(f"Error opening camera capture {self.CAPTURE_SOURCE}: {e_cap}", exc_info=True)
             error_msg = f"Erreur ouverture webcam {self.CAPTURE_SOURCE}: {e_cap}"; self.error_occurred.emit(error_msg)
             self.models_loaded.emit(False); self._running = False
             print(f"DEBUG (VideoThread.run): Exiting run() because camera failed to open.")
             if self.cap: self.cap.release()
             if self.hands_solution: self.hands_solution.close()
             return
        logging.info("Webcam opened successfully.")
        print("DEBUG (VideoThread.run): Webcam opened successfully.")

        sequence_window = deque(maxlen=self.FIXED_LENGTH)
        prediction_display_buffer = deque(maxlen=self.SMOOTHING_WINDOW_SIZE)
        frame_processing_times = deque(maxlen=30); frame_count = 0; last_smoothed_word = "?"
        print("DEBUG (VideoThread.run): Real-time loop variables initialized.")

        target_width = None; target_height = None; resize_needed = False
        try:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width <= 0 or frame_height <= 0: raise ValueError("Invalid frame dimensions from camera")
            logging.info(f"Webcam native resolution: {frame_width}x{frame_height}")
            print(f"DEBUG (VideoThread.run): Native camera resolution: {frame_width}x{frame_height}")
            if self.MAX_FRAME_WIDTH and frame_width > self.MAX_FRAME_WIDTH:
                scale = self.MAX_FRAME_WIDTH / frame_width; target_width = self.MAX_FRAME_WIDTH; target_height = int(frame_height * scale)
                target_height = target_height if target_height % 2 == 0 else target_height + 1; resize_needed = True
                logging.info(f"Display resizing enabled: Target width {target_width}px (height ~{target_height}px)")
                print(f"DEBUG (VideoThread.run): Display will be resized to {target_width}x{target_height}")
            else:
                 target_width = frame_width; target_height = frame_height
                 print("DEBUG (VideoThread.run): No display resizing needed based on MAX_FRAME_WIDTH.")
        except Exception as e_res:
            logging.warning(f"Could not read camera resolution: {e_res}. Using fallback display size.")
            print(f"DEBUG (VideoThread.run): Could not get camera resolution: {e_res}")
            target_width = 640; target_height = 480; resize_needed = True
            print(f"DEBUG (VideoThread.run): Falling back to display size {target_width}x{target_height}")

        print("DEBUG (VideoThread.run): Entering main video loop...")
        loop_count = 0
        while self._running:
            loop_start_time = time.time(); loop_count += 1
            try: ret, frame = self.cap.read()
            except Exception as e_read:
                 logging.error(f"Exception during cap.read() (iteration {loop_count}): {e_read}", exc_info=True)
                 self.error_occurred.emit(f"Erreur lecture webcam: {e_read}"); print(f"DEBUG: Breaking loop... cap.read(): {e_read}"); break
            if not ret or frame is None:
                logging.error(f"Failed to read frame (iteration {loop_count}, ret={ret}, frame is None={frame is None}). Stopping thread.")
                is_still_opened = self.cap.isOpened() if self.cap else False
                error_msg = f"Impossible de lire la frame (tentative {loop_count}). Caméra ouverte: {is_still_opened}"
                if self._running: self.error_occurred.emit(error_msg)
                print(f"DEBUG: Breaking loop... cannot read frame. Camera still opened: {is_still_opened}"); break
            frame_count += 1

            display_frame = None
            if resize_needed and target_width and target_height:
                try: display_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                except Exception as e_resize: logging.warning(f"Error resizing display frame: {e_resize}"); display_frame = frame.copy()
            else: display_frame = frame.copy()

            hands_detected_this_frame = False
            if self.hands_solution:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
                    results = self.hands_solution.process(image_rgb)
                    if results.multi_hand_landmarks:
                        hands_detected_this_frame = True
                        for hand_landmarks in results.multi_hand_landmarks:
                            if self.mp_drawing and self.drawing_spec_hand and self.drawing_spec_connection:
                                try: self.mp_drawing.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.drawing_spec_hand, self.drawing_spec_connection)
                                except Exception as e_draw: logging.warning(f"Error drawing hand landmarks: {e_draw}", exc_info=False); print(f"DEBUG Loop {loop_count}: Warning - error drawing landmarks: {e_draw}")
                except Exception as e_hand_detect: logging.warning(f"Error during Mediapipe hand processing: {e_hand_detect}", exc_info=False); print(f"DEBUG Loop {loop_count}: Warning - error during hand detection: {e_hand_detect}")

            if hands_detected_this_frame != self.last_hands_detected_status:
                if self._running: self.hands_detected_signal.emit(hands_detected_this_frame)
                self.last_hands_detected_status = hands_detected_this_frame; print(f"DEBUG Loop {loop_count}: Hands detected status changed: {hands_detected_this_frame}")

            should_run_inference = False
            hand_check_passed = (not self.hands_solution) or hands_detected_this_frame
            frame_interval_check_passed = (frame_count % (self.FRAMES_TO_SKIP + 1) == 0)
            if hand_check_passed and frame_interval_check_passed: should_run_inference = True

            if should_run_inference:
                inference_start_time = time.time()
                cnn_features = self.extract_cnn_features_realtime(frame) # Appel de la fonction modifiée
                processing_time_ms = (time.time() - inference_start_time) * 1000; frame_processing_times.append(processing_time_ms)
                if cnn_features is not None:
                    sequence_window.append(cnn_features); current_sequence_len = len(sequence_window)
                    if current_sequence_len > 0:
                        padded_sequence = None; current_sequence_np = np.array(sequence_window, dtype=np.float32)
                        if current_sequence_len < self.FIXED_LENGTH:
                            padding_size = self.FIXED_LENGTH - current_sequence_len
                            try:
                                paddings = tf.constant([[padding_size, 0], [0, 0]], dtype=tf.int32)
                                padded_sequence = tf.pad(current_sequence_np, paddings, "CONSTANT", constant_values=0.0).numpy()
                            except Exception as e_pad:
                                 print(f"DEBUG: Error during tf.pad: {e_pad}. Falling back to np.concatenate.")
                                 padding_array = np.zeros((padding_size, self.FEATURE_DIM), dtype=np.float32)
                                 padded_sequence = np.concatenate((padding_array, current_sequence_np), axis=0)
                        else: padded_sequence = current_sequence_np
                        if padded_sequence is not None and padded_sequence.shape == (self.FIXED_LENGTH, self.FEATURE_DIM):
                            reshaped_sequence = np.expand_dims(padded_sequence, axis=0)
                            try:
                                prediction_probs = self.lstm_prediction_model(reshaped_sequence, training=False).numpy()[0]
                                # print(f"DEBUG Loop {loop_count}: Raw prediction probs shape: {prediction_probs.shape}, Max prob: {np.max(prediction_probs):.4f}") # DEBUG
                                top_n_indices = np.argsort(prediction_probs)[-self.TOP_N:][::-1]; top_n_confidences = prediction_probs[top_n_indices]
                                top_n_words = [self.index_to_word.get(idx, f"UNK_{idx}") for idx in top_n_indices]
                                top_n_display_list = [f"{word} ({conf:.2f})" for word, conf in zip(top_n_words, top_n_confidences)]
                                # print(f"DEBUG Loop {loop_count}: Top N Raw: {top_n_display_list}") # DEBUG
                                if self._running: self.top_n_ready.emit(top_n_display_list)
                                top_pred_idx = top_n_indices[0]; top_pred_conf = top_n_confidences[0]
                                # print(f"DEBUG Loop {loop_count}: Top prediction: Idx={top_pred_idx}, Word='{self.index_to_word.get(top_pred_idx, '?')}', Conf={top_pred_conf:.4f}, Threshold={self.PREDICTION_THRESHOLD:.4f}") # DEBUG
                                if top_pred_conf >= self.PREDICTION_THRESHOLD:
                                    prediction_display_buffer.append(top_pred_idx)
                                    # print(f"DEBUG Loop {loop_count}: Prediction added to buffer.") # DEBUG
                                # else: print(f"DEBUG Loop {loop_count}: Prediction skipped (below threshold).") # DEBUG
                            except Exception as e_pred: logging.exception(f"Error during LSTM prediction: {e_pred}"); print(f"DEBUG: Exception during LSTM predict: {e_pred}"); self.top_n_ready.emit(["Erreur Prediction LSTM"])
                        else:
                            if padded_sequence is not None: print(f"DEBUG: Incorrect sequence shape before LSTM: {padded_sequence.shape}")
                            else: print("DEBUG: Padded sequence is None.")
                            self.top_n_ready.emit(["Erreur Shape Séquence"])
                else: print(f"DEBUG Loop {loop_count}: CNN Feature extraction returned None."); self.top_n_ready.emit(["Erreur Extraction CNN"])
            else:
                if self.hands_solution and not hands_detected_this_frame:
                    if sequence_window: sequence_window.clear(); print(f"DEBUG Loop {loop_count}: Hands disappeared/not detected, clearing sequence window.")
                    if prediction_display_buffer:
                        prediction_display_buffer.clear(); print(f"DEBUG Loop {loop_count}: Hands disappeared/not detected, clearing prediction buffer.")
                        if self.last_hands_detected_status:
                             if self._running: self.top_n_ready.emit([""])
                             if last_smoothed_word != "?": last_smoothed_word = "?"; self.prediction_ready.emit(last_smoothed_word)

            current_smoothed_word = "?"
            if prediction_display_buffer:
                try:
                    word_counts = Counter(prediction_display_buffer)
                    most_common_word = word_counts.most_common(1)
                    if most_common_word:
                        smoothed_index = most_common_word[0][0]; current_smoothed_word = self.index_to_word.get(smoothed_index, "?")
                        # print(f"DEBUG Loop {loop_count}: Smoothed word determined: '{current_smoothed_word}' from buffer {list(prediction_display_buffer)}") # DEBUG
                    # else: print(f"DEBUG Loop {loop_count}: Smoothing buffer not empty, but most_common returned nothing.") # DEBUG
                except Exception as e_smooth: logging.warning(f"Error during prediction smoothing: {e_smooth}"); print(f"DEBUG Loop {loop_count}: Exception during smoothing: {e_smooth}")

            if current_smoothed_word != last_smoothed_word:
                # print(f"DEBUG Loop {loop_count}: Emitting smoothed prediction: '{current_smoothed_word}' (Previous: '{last_smoothed_word}')") # DEBUG
                if self._running: self.prediction_ready.emit(current_smoothed_word)
                last_smoothed_word = current_smoothed_word
            # else: print(f"DEBUG Loop {loop_count}: Smoothed word '{current_smoothed_word}' same as last, not emitting.") # DEBUG

            try:
                 if frame_processing_times:
                     avg_proc_time = np.mean(frame_processing_times); fps_proc_approx = 1000 / avg_proc_time if avg_proc_time > 0 else 0
                     cv2.putText(display_frame, f"Proc: {avg_proc_time:.1f}ms (~{fps_proc_approx:.1f} FPS)", (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)
                 loop_time_ms = (time.time() - loop_start_time) * 1000; fps_loop_approx = 1000 / loop_time_ms if loop_time_ms > 0 else 0
                 cv2.putText(display_frame, f"Loop: {loop_time_ms:.1f}ms (~{fps_loop_approx:.1f} FPS)", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.CV_LIGHT_GREY, 1, cv2.LINE_AA)
                 if self.hands_solution:
                     status_text = "Mains: Oui" if hands_detected_this_frame else "Mains: Non"; status_color = Colors.CV_GREEN if hands_detected_this_frame else Colors.CV_RED
                     cv2.putText(display_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
            except Exception as e_display_debug: logging.warning(f"Error drawing debug info: {e_display_debug}", exc_info=False); print(f"DEBUG Loop {loop_count}: Warning - error drawing debug info: {e_display_debug}")

            try:
                 if display_frame is not None and display_frame.size > 0:
                      if self._running: self.frame_ready.emit(display_frame)
                 # else: print(f"DEBUG Loop {loop_count}: display_frame is None or empty, skipping emit.") # DEBUG
            except Exception as e_emit: logging.warning(f"Error emitting frame signal: {e_emit}", exc_info=False); print(f"DEBUG Loop {loop_count}: Warning - error emitting frame: {e_emit}")

        # --- End of Loop ---
        print(f"DEBUG: Exited main video loop after {loop_count} iterations.")
        logging.info("Video thread loop finished or stopped.")
        if self.cap and self.cap.isOpened():
            try: self.cap.release(); logging.info("Webcam released."); print("DEBUG: Camera released.")
            except Exception as e_rel: logging.error(f"Exception releasing camera: {e_rel}")
        else: print("DEBUG: Camera was not open or already released.")
        if self.hands_solution:
            try: self.hands_solution.close(); print("DEBUG: Mediapipe Hands solution closed.")
            except Exception as e_mp_close: print(f"DEBUG: Error closing Mediapipe: {e_mp_close}")
        try: print("DEBUG: Attempting to clear Keras session..."); tf.keras.backend.clear_session(); logging.info("Keras/TensorFlow session cleared."); print("DEBUG: Keras session cleared.")
        except Exception as e_clear: logging.warning(f"Error clearing Keras session: {e_clear}"); print(f"DEBUG: Error clearing Keras session: {e_clear}")
        logging.info("Video thread finished cleanly.")
        print("DEBUG: VideoThread run() finished")

    def stop(self):
        print("DEBUG: VideoThread stop() called"); self._running = False; logging.info("Stop requested for video thread.")


# --- Ui_ParametersWindow Class ---
class Ui_ParametersWindow(QWidget):
    color_changed = Signal(QColor); bg_color_changed = Signal(QColor)
    def setupUi(self, ParametersWindow):
        ParametersWindow.setObjectName(u"ParametersWindow"); ParametersWindow.resize(400, 250); ParametersWindow.setWindowTitle("Paramètres d'Affichage")
        self.main_layout = QVBoxLayout(ParametersWindow); self.main_layout.setSpacing(15); self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.label = QLabel("Paramètres d'Affichage", ParametersWindow); self.label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.label.setStyleSheet(u"font-size: 16px; font-weight: bold; margin-bottom: 10px;"); self.main_layout.addWidget(self.label)
        color_group_box = QFrame(ParametersWindow); color_group_box.setFrameShape(QFrame.Shape.StyledPanel)
        color_layout = QGridLayout(color_group_box); color_layout.setVerticalSpacing(10); color_layout.setHorizontalSpacing(10)
        self.text_color_label = QLabel("Couleur du texte:", color_group_box); self.text_color_btn = QPushButton("Choisir", color_group_box); self.text_color_btn.clicked.connect(self.choose_text_color)
        self.text_color_preview = QLabel(color_group_box); self.text_color_preview.setFixedSize(25, 25); self.text_color_preview.setStyleSheet("background-color: white; border: 1px solid black; border-radius: 3px;"); self.text_color_preview.setToolTip("Aperçu couleur texte")
        self.bg_color_label = QLabel("Couleur de fond:", color_group_box); self.bg_color_btn = QPushButton("Choisir", color_group_box); self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.bg_color_preview = QLabel(color_group_box); self.bg_color_preview.setFixedSize(25, 25); self.bg_color_preview.setStyleSheet("background-color: rgb(10, 32, 77); border: 1px solid black; border-radius: 3px;"); self.bg_color_preview.setToolTip("Aperçu couleur fond")
        color_layout.addWidget(self.text_color_label, 0, 0); color_layout.addWidget(self.text_color_preview, 0, 1); color_layout.addWidget(self.text_color_btn, 0, 2)
        color_layout.addWidget(self.bg_color_label, 1, 0); color_layout.addWidget(self.bg_color_preview, 1, 1); color_layout.addWidget(self.bg_color_btn, 1, 2)
        color_layout.setColumnStretch(0, 1); color_layout.setColumnStretch(2, 0); self.main_layout.addWidget(color_group_box); self.main_layout.addStretch(1)
        self.buttons_layout = QHBoxLayout(); self.buttons_layout.addStretch(1); self.default_btn = QPushButton("Par défaut", ParametersWindow); self.default_btn.setToolTip("Réinitialiser les couleurs par défaut"); self.default_btn.clicked.connect(self.reset_defaults); self.buttons_layout.addWidget(self.default_btn)
        self.close_btn = QPushButton("Fermer", ParametersWindow); self.close_btn.clicked.connect(ParametersWindow.close); self.buttons_layout.addWidget(self.close_btn); self.main_layout.addLayout(self.buttons_layout); ParametersWindow.setLayout(self.main_layout)
    @Slot()
    def choose_text_color(self):
        parent = self.parentWidget() if self.parentWidget() else self; current = self.text_color_preview.palette().window().color(); color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur du texte")
        if color.isValid(): self.text_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;"); self.color_changed.emit(color); print(f"DEBUG (Ui_ParametersWindow): Text color chosen: {color.name()}")
    @Slot()
    def choose_bg_color(self):
        parent = self.parentWidget() if self.parentWidget() else self; current = self.bg_color_preview.palette().window().color(); color = QColorDialog.getColor(current, parent=parent, title="Choisir couleur de fond")
        if color.isValid(): self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black; border-radius: 3px;"); self.bg_color_changed.emit(color); print(f"DEBUG (Ui_ParametersWindow): Background color chosen: {color.name()}")
    @Slot()
    def reset_defaults(self):
        default_text = QColor("white"); default_bg = QColor(10, 32, 77)
        self.text_color_preview.setStyleSheet(f"background-color: {default_text.name()}; border: 1px solid black; border-radius: 3px;"); self.bg_color_preview.setStyleSheet(f"background-color: {default_bg.name()}; border: 1px solid black; border-radius: 3px;")
        self.color_changed.emit(default_text); self.bg_color_changed.emit(default_bg); print("DEBUG (Ui_ParametersWindow): Colors reset to default.")


# --- Ui_MainWindow Class (Structure verticale) ---
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(800, 750) # Ajuster la taille par défaut si besoin
        MainWindow.setWindowTitle("Traduction LSF en Temps Réel (Sans Audio)")

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

        # --- Rangée 3: Contrôles (Export) ---
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

    def setup_top_toolbar(self, parent):
        self.gridLayout_top_toolbar = QGridLayout()
        self.gridLayout_top_toolbar.setObjectName(u"gridLayout_TopToolbar")
        self.boutonparametre = QPushButton(parent); self.boutonparametre.setObjectName(u"boutonparametre"); self.boutonparametre.setFixedSize(QSize(50, 50)); self.boutonparametre.setToolTip("Ouvrir les paramètres d'affichage"); self.boutonparametre.setText("⚙️"); self.boutonparametre.setFont(QFont("Segoe UI Emoji", 16))
        self.boutonparametre.setStyleSheet("QPushButton {border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white;} QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); }")
        self.gridLayout_top_toolbar.addWidget(self.boutonparametre, 0, 0, Qt.AlignmentFlag.AlignLeft)
        self.logo = QLabel(parent); self.logo.setObjectName(u"logo"); self.logo.setText("Traduction LSF"); self.logo.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: transparent;"); self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayout_top_toolbar.addWidget(self.logo, 0, 1)
        spacerItem = QSpacerItem(50, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum); self.gridLayout_top_toolbar.addItem(spacerItem, 0, 2)
        self.gridLayout_top_toolbar.setColumnStretch(1, 1)

    def setup_camera_view(self, parent):
        self.frame_camera = QFrame(parent)
        self.frame_camera.setObjectName(u"frame_camera")
        sizePolicyCamFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_camera.setSizePolicy(sizePolicyCamFrame)
        self.frame_camera.setMinimumHeight(300)
        self.frame_camera.setStyleSheet("QFrame#frame_camera { border: 1px solid gray; border-radius: 5px; background-color: black; }")
        gridLayout_camera_inner = QGridLayout(self.frame_camera)
        gridLayout_camera_inner.setObjectName(u"gridLayout_camera_inner")
        gridLayout_camera_inner.setContentsMargins(1, 1, 1, 1)
        self.camera_view = QLabel(self.frame_camera)
        self.camera_view.setObjectName(u"camera_view")
        sizePolicyCamLabel = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera_view.setSizePolicy(sizePolicyCamLabel)
        self.camera_view.setStyleSheet(u"QLabel#camera_view { background-color: transparent; border: none; color: grey; font-size: 16pt; }")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setScaledContents(False)
        gridLayout_camera_inner.addWidget(self.camera_view, 0, 0, 1, 1)

    def setup_text_area(self, parent):
        self.frame_text = QFrame(parent)
        self.frame_text.setObjectName(u"frame_text")
        sizePolicyTextFrame = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_text.setSizePolicy(sizePolicyTextFrame)
        self.frame_text.setMinimumHeight(100)
        self.frame_text.setMaximumHeight(250)
        self.frame_text.setStyleSheet("QFrame#frame_text { background-color: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 5px; padding: 5px; }")
        layout_inside_frame = QVBoxLayout(self.frame_text)
        layout_inside_frame.setContentsMargins(5, 5, 5, 5)
        layout_inside_frame.setSpacing(5)
        self.label_predictions = QLabel("Prédictions:", self.frame_text)
        self.label_predictions.setObjectName(u"label_predictions")
        font_pred = QFont(); font_pred.setPointSize(11); font_pred.setBold(True); self.label_predictions.setFont(font_pred)
        self.label_predictions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.label_predictions.setStyleSheet("background-color: transparent; color: white;")
        layout_inside_frame.addWidget(self.label_predictions)
        self.textEdit = QTextEdit(self.frame_text)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setStyleSheet(u"QTextEdit { font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: white; }")
        self.textEdit.setReadOnly(True)
        layout_inside_frame.addWidget(self.textEdit, 1)

    def setup_export_controls(self, parent):
        self.horizontalLayout_export = QHBoxLayout()
        self.horizontalLayout_export.setObjectName(u"horizontalLayout_export")
        self.verticalLayout_export = QVBoxLayout()
        self.verticalLayout_export.setObjectName(u"verticalLayout_export")
        self.exportation = QPushButton(parent)
        self.exportation.setObjectName(u"exportation")
        self.exportation.setFixedSize(QSize(50, 50))
        self.exportation.setText("💾"); self.exportation.setFont(QFont("Segoe UI Emoji", 16))
        self.exportation.setToolTip("Exporter le texte (Non implémenté)"); self.exportation.setEnabled(False)
        self.exportation.setStyleSheet("QPushButton { border-radius: 25px; background-color: rgba(255, 255, 255, 0.1); color: white; border: 1px solid rgba(255, 255, 255, 0.3); } QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); } QPushButton:pressed { background-color: rgba(255, 255, 255, 0.3); } QPushButton:disabled { background-color: rgba(128, 128, 128, 0.2); color: gray; border-color: rgba(128, 128, 128, 0.4); }")
        self.verticalLayout_export.addWidget(self.exportation)
        self.horizontalLayout_export.addLayout(self.verticalLayout_export)

    def setup_menu_statusbar(self, MainWindow):
        self.statusbar = QStatusBar(MainWindow); self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setStyleSheet("QStatusBar { color: #DDDDDD; padding-left: 5px; background-color: transparent; }")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow); self.menubar.setObjectName(u"menubar")
        MainWindow.setMenuBar(self.menubar)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        if hasattr(self, 'boutonparametre'): self.boutonparametre.setToolTip(_translate("MainWindow", u"Ouvrir les paramètres d'affichage", None))
        if hasattr(self, 'exportation'): self.exportation.setToolTip(_translate("MainWindow", u"Exporter le texte (Non implémenté)", None))
        if hasattr(self, 'logo') and self.logo.text() == "": self.logo.setText(_translate("MainWindow", u"Traduction LSF", None))
        if hasattr(self, 'camera_view'): self.camera_view.setText(_translate("MainWindow", u"Initialisation...", None))
        if hasattr(self, 'textEdit'): self.textEdit.setPlaceholderText(_translate("MainWindow", u"Les mots prédits apparaîtront ici...", None))
        if hasattr(self, 'label_predictions'): self.label_predictions.setText(_translate("MainWindow", u"Prédictions :", None))
        # if hasattr(self, 'label_export'): self.label_export.setText(_translate("MainWindow", u"Exporter", None))


# --- MainWindow Class (Application Logic) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        print("DEBUG (MainWindow.__init__): UI setup complete (Vertical Layout).")

        self.parameters_window = None; self.ui_parameters = None
        self.ui.boutonparametre.clicked.connect(self.open_parameters)
        print("DEBUG (MainWindow.__init__): Parameters button connected.")

        self.current_text_color = self.ui.default_text_color
        self.current_bg_color = self.ui.default_bg_color
        self.apply_initial_styles()
        print("DEBUG (MainWindow.__init__): Initial styles applied.")

        self.placeholder_timer = QTimer(self); self.placeholder_timer.timeout.connect(self.show_placeholder_frame)
        self.placeholder_frame_counter = 0; self.placeholder_active = True
        self.ui.camera_view.setText("Initialisation..."); self.ui.camera_view.setAlignment(Qt.AlignCenter)
        print("DEBUG (MainWindow.__init__): Placeholder timer setup.")

        self.video_thread = VideoThread(self)
        print("DEBUG (MainWindow.__init__): VideoThread instance created.")
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.prediction_ready.connect(self.update_prediction)
        self.video_thread.top_n_ready.connect(self.update_top_n_status)
        self.video_thread.error_occurred.connect(self.handle_error)
        self.video_thread.models_loaded.connect(self.on_models_loaded)
        self.video_thread.finished.connect(self.on_thread_finished)
        self.video_thread.hands_detected_signal.connect(self.update_hand_detection_status)
        print("DEBUG (MainWindow.__init__): VideoThread signals connected.")

        self.ui.statusbar.showMessage("Initialisation: Chargement des modèles et de la caméra...")
        self.placeholder_timer.start(50)
        print("DEBUG (MainWindow.__init__): Placeholder timer started.")
        print("DEBUG (MainWindow.__init__): Starting VideoThread...")
        self.video_thread.start()

    def apply_initial_styles(self):
         print("DEBUG (MainWindow.apply_initial_styles): Applying initial styles.")
         bg_color_name = self.current_bg_color.name()
         style = f"QWidget#centralwidget {{ background-color: {bg_color_name}; }}"
         self.ui.centralwidget.setStyleSheet(style)
         self.update_text_colors(self.current_text_color)

    @Slot()
    def show_placeholder_frame(self):
        if not self.placeholder_active: return
        try:
            label = self.ui.camera_view
            if not label or not label.isVisible() or label.width() <= 0 or label.height() <= 0: return
            w = label.width(); h = label.height(); pixmap = QPixmap(w, h); pixmap.fill(Qt.black)
            painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            center_x = w / 2.0; center_y = h / 2.0; max_radius = min(w, h) / 7.0; radius_variation = max_radius / 2.5
            pulse = (1 + np.sin(self.placeholder_frame_counter * 0.15)) / 2.0; current_radius = max_radius - (radius_variation * pulse)
            if current_radius > 0: painter.setBrush(QColor(50, 50, 50)); painter.setPen(Qt.NoPen); painter.drawEllipse(QPointF(center_x, center_y), current_radius, current_radius)
            font = QFont("Arial", 14); painter.setFont(font); painter.setPen(QColor(180, 180, 180))
            text = "En attente de la caméra..."
            if self.video_thread and self.video_thread.isFinished():
                 current_status = self.ui.statusbar.currentMessage()
                 if "ERREUR" in current_status or "ÉCHEC" in current_status: text = "Échec initialisation / Erreur"
                 else: text = "Caméra déconnectée"
            painter.drawText(pixmap.rect(), Qt.AlignCenter, text); painter.end(); label.setPixmap(pixmap)
            self.placeholder_frame_counter += 1
        except Exception as e:
             print(f"ERROR (MainWindow.show_placeholder_frame): {e}")
             if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
             self.placeholder_active = False
             if label: label.setText(f"Erreur Placeholder:\n{e}")

    @Slot(np.ndarray)
    def update_frame(self, cv_img):
        if self.placeholder_active:
            print("DEBUG (MainWindow.update_frame): First real frame received, stopping placeholder.")
            if self.placeholder_timer.isActive(): self.placeholder_timer.stop()
            self.placeholder_active = False; self.ui.camera_view.clear(); self.ui.camera_view.setText("")
        if cv_img is None or cv_img.size == 0: print("DEBUG (MainWindow.update_frame): Received empty/invalid frame, skipping."); return
        try:
            h, w, ch = cv_img.shape; bytes_per_line = ch * w
            qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            if qt_image.isNull(): print("ERROR DEBUG (MainWindow.update_frame): QImage creation failed!"); self.ui.camera_view.setText("Erreur: QImage Null"); return
            qt_pixmap = QPixmap.fromImage(qt_image)
            if qt_pixmap.isNull(): print("ERROR DEBUG (MainWindow.update_frame): QPixmap conversion failed!"); self.ui.camera_view.setText("Erreur: QPixmap Null"); return
            label = self.ui.camera_view; label_size = label.size()
            if label_size.isValid() and label_size.width() > 10 and label_size.height() > 10:
                scaled_pixmap = qt_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if scaled_pixmap.isNull(): print("ERROR DEBUG (MainWindow.update_frame): Scaled QPixmap is Null!"); label.setPixmap(qt_pixmap)
                else: label.setPixmap(scaled_pixmap)
            else: label.setPixmap(qt_pixmap)
        except cv2.error as e_cv:
             logging.error(f"OpenCV error in update_frame: {e_cv}", exc_info=True); print(f"DEBUG (MainWindow.update_frame): OpenCV error: {e_cv}")
             self.ui.camera_view.setText(f"Erreur OpenCV:\n{e_cv}")
             if not self.placeholder_active: self.placeholder_active = True; self.placeholder_timer.start(50)
        except Exception as e:
            logging.error(f"Error updating frame: {e}", exc_info=True); print(f"DEBUG (MainWindow.update_frame): Exception: {e}")
            self.ui.camera_view.setText(f"Erreur affichage:\n{str(e)}")
            if not self.placeholder_active: self.placeholder_active = True; self.placeholder_timer.start(50)

    @Slot(str)
    def update_prediction(self, text):
        if text and text != "?":
            current_content = self.ui.textEdit.toPlainText(); words = current_content.split()
            if not words or text.lower() != words[-1].lower():
                 max_words_history = 50; new_words = words + [text]
                 if len(new_words) > max_words_history: new_words = new_words[-max_words_history:]
                 self.ui.textEdit.setPlainText(" ".join(new_words)); self.ui.textEdit.moveCursor(QTextCursor.MoveOperation.End)

    @Slot(list)
    def update_top_n_status(self, top_n_list):
        filtered_list = [item for item in top_n_list if item and item.strip()]
        if filtered_list:
            status_text = " | ".join(filtered_list[:self.video_thread.ui_TOP_N])
            self.ui.statusbar.showMessage(status_text)
        else:
            current_message = self.ui.statusbar.currentMessage()
            if not current_message or ("ERREUR" not in current_message and "ÉCHEC" not in current_message):
                 self.ui.statusbar.showMessage("Prêt.", 3000)

    @Slot(bool)
    def update_hand_detection_status(self, detected): pass

    @Slot(str)
    def handle_error(self, message):
        logging.error(f"Error reported from video thread: {message}"); print(f"DEBUG (MainWindow.handle_error): Received error signal: {message}")
        critical_keywords = ["impossible d'ouvrir", "webcam", "fichier modèle", "vocabulaire", "shape lstm", "erreur chargement cnn", "erreur chargement lstm", "erreur ouverture webcam", "erreur lecture webcam", "mediapipe", "config gpu", "critical", "manquant", "not found", "empty", "failed to initialize", "attributeerror", "indexerror", "valueerror", "runtimeerror"]
        is_critical = any(keyword in message.lower() for keyword in critical_keywords)
        if is_critical:
             print(f"DEBUG (MainWindow.handle_error): Critical error identified: {message}")
             self.ui.statusbar.showMessage(f"ERREUR CRITIQUE: {message}", 0); self.ui.camera_view.setText(f"ERREUR CRITIQUE:\n{message}\nVérifiez la console/logs.")
             if not self.placeholder_active: print("DEBUG (MainWindow.handle_error): Reactivating placeholder..."); self.placeholder_active = True; self.placeholder_timer.start(50)
             QMessageBox.critical(self, "Erreur Critique", f"Une erreur critique est survenue :\n\n{message}\n\nLa traduction risque de ne pas fonctionner correctement.\nVeuillez vérifier la configuration et les logs.")
        else: self.ui.statusbar.showMessage(f"Erreur: {message}", 10000)

    @Slot(bool)
    def on_models_loaded(self, success):
        print(f"DEBUG (MainWindow.on_models_loaded): Received models_loaded signal with success={success}")
        if success: self.ui.statusbar.showMessage("Modèles chargés. Démarrage de la capture webcam...", 5000)
        else:
            final_fail_msg = "ÉCHEC INITIALISATION: Modèles/Vocab/Webcam. Vérifiez logs."
            self.ui.statusbar.showMessage(final_fail_msg, 0); self.ui.camera_view.setText(f"ÉCHEC INITIALISATION:\nVérifiez la console/logs.")
            print(f"DEBUG (MainWindow.on_models_loaded): Initialization failed (models_loaded=False received)")
            if not self.placeholder_active: self.placeholder_active = True; self.placeholder_timer.start(50)

    @Slot()
    def on_thread_finished(self):
        logging.info("Video processing thread has finished execution."); print("DEBUG (MainWindow.on_thread_finished): Video thread finished signal received.")
        current_status = self.ui.statusbar.currentMessage()
        if "ERREUR" not in current_status and "ÉCHEC" not in current_status: self.ui.statusbar.showMessage("Connexion caméra terminée.", 5000)
        if not self.placeholder_active: print("DEBUG (MainWindow.on_thread_finished): Reactivating placeholder."); self.placeholder_active = True; self.placeholder_timer.start(50)
        if "ERREUR" in current_status or "ÉCHEC" in current_status: self.ui.camera_view.setText("Caméra déconnectée / Erreur")
        else: self.ui.camera_view.setText("Caméra déconnectée")

    def open_parameters(self):
        print("DEBUG (MainWindow.open_parameters): Opening parameters window...")
        if self.parameters_window is None:
            self.parameters_window = QWidget(self, Qt.WindowType.Window); self.parameters_window.setWindowModality(Qt.WindowModality.NonModal)
            self.ui_parameters = Ui_ParametersWindow(); self.ui_parameters.setupUi(self.parameters_window)
            if self.ui_parameters:
                self.ui_parameters.text_color_preview.setStyleSheet(f"background-color: {self.current_text_color.name()}; border: 1px solid black; border-radius: 3px;")
                self.ui_parameters.bg_color_preview.setStyleSheet(f"background-color: {self.current_bg_color.name()}; border: 1px solid black; border-radius: 3px;")
                self.ui_parameters.color_changed.connect(self.update_text_colors); self.ui_parameters.bg_color_changed.connect(self.update_bg_color)
                print("DEBUG (MainWindow.open_parameters): Parameters window created and signals connected.")
            else: print("ERROR: Failed to setup parameters window UI."); self.parameters_window = None; return
        else: print("DEBUG (MainWindow.open_parameters): Parameters window already exists.")
        self.parameters_window.show(); self.parameters_window.raise_(); self.parameters_window.activateWindow()

    @Slot(QColor)
    def update_text_colors(self, color):
        if not color.isValid(): return
        self.current_text_color = color; color_name = color.name(); print(f"DEBUG (MainWindow.update_text_colors): Applying text color: {color_name}")
        text_style = f"color: {color_name}; background-color: transparent;"; bold_text_style = f"color: {color_name}; background-color: transparent; font-weight: bold;"
        if hasattr(self.ui, 'label_export'): self.ui.label_export.setStyleSheet(bold_text_style)
        if hasattr(self.ui, 'label_predictions'): self.ui.label_predictions.setStyleSheet(bold_text_style)
        if hasattr(self.ui, 'logo'): self.ui.logo.setStyleSheet(f"font-size: 24px; {bold_text_style}")
        if hasattr(self.ui, 'statusbar'): self.ui.statusbar.setStyleSheet(f"QStatusBar {{ color: {color.lighter(110).name()}; padding-left: 5px; background-color: transparent; }}")
        if hasattr(self.ui, 'textEdit'): self.ui.textEdit.setStyleSheet(f"QTextEdit {{ font: 14pt 'Segoe UI'; background-color: transparent; border: none; color: {color_name}; }}")

    @Slot(QColor)
    def update_bg_color(self, color):
        if not color.isValid(): return
        self.current_bg_color = color; color_name = color.name(); print(f"DEBUG (MainWindow.update_bg_color): Applying background color: {color_name}")
        style = f"QWidget#centralwidget {{ background-color: {color_name}; }}"
        self.ui.centralwidget.setStyleSheet(style)
        self.update_text_colors(self.current_text_color) # Réappliquer couleur texte

    def closeEvent(self, event):
        logging.info("Main window close requested."); print("DEBUG (MainWindow.closeEvent): Close event triggered.")
        if self.placeholder_timer.isActive(): print("DEBUG: Stopping placeholder timer."); self.placeholder_timer.stop()
        if self.parameters_window and self.parameters_window.isVisible(): print("DEBUG: Closing parameters window."); self.parameters_window.close()
        if self.video_thread and self.video_thread.isRunning():
            logging.info("Stopping video thread..."); print("DEBUG: Signaling video thread to stop...")
            self.video_thread.stop(); print("DEBUG: Waiting for video thread...")
            if not self.video_thread.wait(3000): logging.warning("Video thread timeout."); print("DEBUG: Video thread timed out.")
            else: logging.info("Video thread stopped."); print("DEBUG: Video thread stopped.")
        else: logging.info("Video thread not running."); print("DEBUG: Video thread not running.")
        print("DEBUG (MainWindow.closeEvent): Accepting close event."); event.accept()

# --- Application Entry Point ---
if __name__ == "__main__":
     print("DEBUG: Application starting execution from mainwindow.py")
     app = QApplication(sys.argv)
     window = MainWindow()
     window.showMaximized() # Ou window.show()
     print("DEBUG: Entering Qt application event loop...")
     exit_code = app.exec()
     print(f"DEBUG: Application event loop finished with exit code {exit_code}.")
     sys.exit(exit_code)