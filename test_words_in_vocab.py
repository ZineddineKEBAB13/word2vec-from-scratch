from gensim.models import KeyedVectors

w2v = KeyedVectors.load(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\model.kv", mmap='r')

def check_vocab(filename, model):
    print(f"\nVérification du fichier : {filename}")
    total_analogies = 0
    missing = set()

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # sauter les lignes vides et les commentaires
            if not line or line.startswith("#"):
                continue

            words = line.split()
            # normalement on a 4 mots : a b c d
            total_analogies += 1
            for w in words:
                if w not in model.key_to_index:
                    missing.add(w)

    print(f"Nombre total d'analogies : {total_analogies}")
    print(f"Mots absents du vocabulaire : {len(missing)}")
    if missing:
        print("Exemples :", list(missing)[:20])
    else:
        print("Tous les mots sont présents dans le modèle")
    return missing

# Vérification des deux fichiers
missing_sem = check_vocab(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\semantique.txt", w2v)
missing_syn = check_vocab(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\syntaxique.txt", w2v)
