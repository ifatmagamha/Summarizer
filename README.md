# Bonjour Mme,

Le contenu des fichiers (`train.py`, `model.py`, `utils.py`, `app.py`) est le même que le notebook Colab.

La séparation en fichiers distincts a été faite uniquement pour permettre la création d’une interface **Streamlit** et tester le résumé automatique de manière interactive.

## Contenu du dossier

- **train.py** : entraînement du modèle RNN + attention et sauvegarde du vocabulaire et des poids dans `saved_models/`.
- **model.py** : définitions des classes `EncoderRNN` et `SentenceSelector`.
- **utils.py** : fonctions utilitaires pour le prétraitement, la tokenization, TF-IDF, encodage/décodage.
- **app.py** : interface Streamlit pour tester le résumé automatique.
- **saved_models/** : modèles et vocabulaire sauvegardés après exécution de `train.py`.


## Instructions pour lancer l'application

1. Installer les dépendances 
2. Lancer l'application

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
---


Merci pour votre compréhension.
