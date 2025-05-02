import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization # Importer les couches nécessaires

# --- Constants (Rappel) ---
FIXED_LENGTH = 46
FEATURES_PER_FRAME = 12 # 4 points * 3 coords

def create_sequential_model(input_shape, num_classes):
    """Crée un modèle basé sur LSTM pour la reconnaissance de séquences."""
    # input_shape = (FIXED_LENGTH, FEATURES_PER_FRAME), ex: (46, 12)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='lstm_input'), # Shape (None, 46, 12)

        # --- Couches LSTM ---
        # Optionnel: Normalisation par batch sur la dimension temporelle si utile
        # tf.keras.layers.BatchNormalization(axis=-1), # Normalise les features sur chaque pas de temps

        tf.keras.layers.LSTM(64, return_sequences=True, name='lstm_1'),
        # Optionnel: BatchNormalization après LSTM (parfois utile)
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),

        tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2'), # Une autre couche LSTM
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),

        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_3'), # Dernière LSTM, return_sequences=False
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3, name='dropout_3'),

        # --- Couches Dense pour Classification ---
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.4, name='dropout_4'), # Dropout potentiellement plus élevé avant la sortie

        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer') # Couche de sortie finale
    ])
    return model

