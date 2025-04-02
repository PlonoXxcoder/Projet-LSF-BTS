# Traducteur LSF vers Texte et Audio - Projet BTS CIEL 2024/2025

Ce dépôt contient une partie du projet de BTS CIEL (Cybersécurité, Informatique et réseaux, Électronique) visant à développer un système de traduction de la Langue des Signes Française (LSF) vers du texte et de l'audio. L'objectif principal est de faciliter la communication pour les personnes malentendantes et leur entourage.Le système comprend plusieurs modules: capture vidéo, reconnaissance des signes, traduction, et synthèse vocale.


**Contexte du projet :**
*   **Formation :** BTS CIEL - Lycée Edouard Branly
*   **Épreuve :** E6
*   **Durée :** 150 heures
*   **Équipe :** 4 étudiants (voir section Rôles)
*   **Public Cible :** Personnes malentendantes et leur entourage

## Objectifs Généraux du Projet (selon Cahier des Charges)

1.  **Développer un système de capture vidéo** et de traitement d'image pour détecter les mains et le visage.
2.  **Implémenter un module d'Intelligence Artificielle** pour la reconnaissance des signes LSF capturés.
3.  **Créer un module de traduction** pour convertir les signes reconnus en phrases françaises textuelles, en gérant la grammaire et la syntaxe.
4.  **Intégrer un module de synthèse vocale** pour convertir le texte traduit en parole audible.
5.  **Concevoir une interface utilisateur graphique (GUI)** accessible et intuitive.
6.  **Assurer une traduction en temps réel ou quasi-réel** (latence < 2 secondes).
7.  **Atteindre une précision de reconnaissance supérieure à 85%.**

## Installation

## Technologies Utilisées

*   **Python 3.18.10**
*   **TensorFlow / Keras:** Framework principal pour la construction et l'entraînement du modèle LSTM.
*   **MediaPipe:** (Utilisé en amont) Pour l'extraction des points clés du corps et des mains à partir des vidéos. Ce script *ne fait pas* l'extraction, il utilise les résultats.
*   **OpenCV (`opencv-python`):** Utilisé potentiellement pour le traitement vidéo en amont (non présent dans ce script).
*   **NumPy:** Manipulation efficace des tableaux numériques (séquences de points clés).
*   **Scikit-learn:** Pour la division des données (`train_test_split`).
*   **imbalanced-learn:** (Optionnel) Pour l'équilibrage des classes avec SMOTE.
*   **Keras Tuner:** Pour l'optimisation automatique des hyperparamètres.
*   **Matplotlib:** Pour la visualisation des graphiques d'entraînement.


## Workflow Global du Système (Tel que Prévu)

1.  **Capture :** La caméra filme l'utilisateur signant (Module Étudiant 1).
2.  **Extraction :** MediaPipe détecte et extrait les coordonnées (x, y, z) des points clés des mains et du corps pour chaque image (Module Étudiant 1).
3.  **Normalisation/Prétraitement :** Les séquences de points clés sont normalisées et formatées (Module Étudiant 1 ou 2).
4.  **Reconnaissance :** La séquence prétraitée est envoyée au modèle IA entraîné (ce dépôt, Module Étudiant 2) qui prédit le signe LSF correspondant.
5.  **Traduction :** Le signe (ou la séquence de signes) reconnu est traduit en texte français par un module dédié (potentiellement avec gestion grammaticale) (Module externe au focus de ce dépôt).
6.  **Synthèse Vocale :** Le texte français est converti en audio (Module Étudiant 4).
7.  **Affichage :** L'interface utilisateur (Module Étudiant 3) affiche la vidéo en direct, le texte traduit et potentiellement joue l'audio.

## Workflow Global du Système

1.  **Capture :** La caméra filme l'utilisateur signant (Module Étudiant 1).
2.  **Extraction :** MediaPipe détecte et extrait les coordonnées (x, y, z) des points clés des mains et du corps pour chaque image (Module Étudiant 1).
3.  **Normalisation/Prétraitement :** Les séquences de points clés sont normalisées et formatées (Module Étudiant 1 ou 2).
4.  **Reconnaissance :** La séquence prétraitée est envoyée au modèle IA entraîné (ce dépôt, Module Étudiant 2) qui prédit le signe LSF correspondant.
5.  **Traduction :** Le signe (ou la séquence de signes) reconnu est traduit en texte français par un module dédié (potentiellement avec gestion grammaticale) (Module externe au focus de ce dépôt).
6.  **Synthèse Vocale :** Le texte français est converti en audio (Module Étudiant 4).
7.  **Affichage :** L'interface utilisateur (Module Étudiant 3) affiche la vidéo en direct, le texte traduit et potentiellement joue l'audio.

## Équipe et Répartition des Rôles

*   **@PlonoXxcoder :** Responsable du module de capture vidéo et traitement d'image.
*   **@PlonoXxcoder et @aj69210 :** Responsable du développement IA et reconnaissance des signes (**Focus de ce dépôt**).
*   **@Walid01100 :** Responsable de l'interface utilisateur et de l'expérience utilisateur.
*   **@Rafael1101001 :** Responsable de la synthèse vocale et de l'intégration audio.

*Chaque personne est responsable de la documentation, des tests unitaires/d'intégration de sa partie et de la contribution à la présentation finale.*

## Coordination de l'Équipe

*   Réunions hebdomadaires de suivi.
*   Documentation collaborative sur Wiki.

