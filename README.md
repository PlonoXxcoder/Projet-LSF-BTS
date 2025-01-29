# Projet-LSF-BTS

Projet de traducteur LSF (Langue des Signes Française) vers texte et audio pour le BTS CIEL.

## Description

Ce projet vise à développer un système de traduction de la Langue des Signes Française (LSF) en texte et audio. Le système comprend plusieurs modules: capture vidéo, reconnaissance des signes, traduction, et synthèse vocale.

## Structure du projet

- `capture_video/`: Module de capture et traitement d'image.
- `reconnaissance_LSF/`: Module de reconnaissance des signes LSF.
- `traduction/`: Module de traduction des signes en texte français.
- `synthese_vocale/`: Module de conversion texte vers parole.
- `interface/`: Interface utilisateur.
- `tests/`: Tests unitaires et d'intégration.
- `docs/`: Documentation du projet.

## Installation

```bash
git clone <URL_DU_DEPOT>
cd Projet-LSF-BTS
python -m venv venv
source venv/bin/activate  # Sur Windows, utiliser `venv\Scripts\activate`
pip install -r requirements.txt
