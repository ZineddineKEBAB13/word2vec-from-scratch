# Word2Vec from scratch (Skip-gram + Negative Sampling)

Implémentation de Word2Vec en Python/NumPy pour apprendre des embeddings statiques à partir d’un corpus tokenisé.
Le projet inclut une version baseline, une version améliorée, une évaluation par triplets (similarité cosinus) et une évaluation par analogies sémantiques et syntaxiques.

## Fonctionnalités
- Construction du vocabulaire avec filtrage par fréquence (min_count)
- Génération de paires centre–contexte via fenêtre de contexte
- Entraînement Skip-gram avec Negative Sampling
- Version améliorée avec :
  - distribution de tirage négatif basée sur unigram puissance 0.75
  - table de sampling pour accélérer les tirages négatifs
  - subsampling des mots fréquents
  - décroissance progressive du learning rate
- Export des embeddings au format texte compatible KeyedVectors
- Évaluation :
  - score sur triplets en similarité cosinus
  - analogies b - a + c avec recherche de voisins accélérée par KDTree

## Structure du projet
- `w2v.py` : version baseline Word2Vec SGNS
- `w2vAmel.py` : version améliorée Word2Vec SGNS
- `create_vocab.py` : création du vocabulaire et des mappings
- `eval_w2v.py` : évaluation par triplets (cosine similarity)
- `analogies_w2v.py` : évaluation par analogies avec KDTree
- `test_words_in_vocab.py` : vérification des mots présents dans le vocabulaire
- `tp_w2v.pdf` : consignes du TP
- `semantique.txt` : analogies sémantiques
- `syntaxique.txt` : analogies syntaxiques
- `resultats_semantique.txt` : sorties analogies sémantiques
- `resultats_syntaxique.txt` : sorties analogies syntaxiques
- `Le_comte_de_Monte_Cristo.tok` : corpus tokenisé d’entraînement

## Prérequis
- Python 3.10+
- Dépendances principales : numpy, scipy, gensim
