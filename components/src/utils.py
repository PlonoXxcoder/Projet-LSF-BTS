import numpy as np
import cv2

def normalize_keypoints(keypoints, image_width, image_height):
    """Normalise les keypoints en les divisant par la taille de l'image."""
    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, :, 0] /= image_width
    normalized_keypoints[:, :, 1] /= image_height
    normalized_keypoints[:, :, 2] /= max(image_width, image_height)  # Normaliser aussi la profondeur
    return normalized_keypoints

def augment_data(keypoints, rotation_range=(-10, 10), translation_range=(-0.1, 0.1), scaling_range=(0.9, 1.1)):
    """Augmente les données en appliquant des rotations, des translations et des scalings aléatoires."""
    augmented_keypoints = keypoints.copy()

    # Rotation
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                                [0, 0, 1]])
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            augmented_keypoints[i, j] = np.dot(rotation_matrix, augmented_keypoints[i, j])

    # Translation
    translation_x = np.random.uniform(translation_range[0], translation_range[1])
    translation_y = np.random.uniform(translation_range[0], translation_range[1])
    augmented_keypoints[:, :, 0] += translation_x
    augmented_keypoints[:, :, 1] += translation_y

    # Scaling
    scale = np.random.uniform(scaling_range[0], scaling_range[1])
    augmented_keypoints[:, :, 0] *= scale
    augmented_keypoints[:, :, 1] *= scale
    augmented_keypoints[:, :, 2] *= scale

    return augmented_keypoints

def visualize_keypoints(frame, keypoints, color=(0, 255, 0), radius=3):
    """Visualise les keypoints sur une image."""
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            x, y, z = keypoints[i, j]
            x = int(x)
            y = int(y)
            cv2.circle(frame, (x, y), radius, color, -1)
    return frame

def load_video_frames(video_path, target_resolution=(64, 64)):
    """Charge les frames d'une vidéo et les redimensionne à la résolution cible."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_resolution)
        frames.append(frame)

    cap.release()
    return np.array(frames)