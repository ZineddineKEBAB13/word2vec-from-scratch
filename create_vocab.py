from gensim.models import KeyedVectors

###  un petit code initial pour tranformer model.bin en model.kv afin d'avoir un acces plus rapide au donnees ##
#BIN_PATH = r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\model.bin"
#w2v = KeyedVectors.load_word2vec_format(BIN_PATH, binary=True)
#w2v.save("model.kv")


w2v = KeyedVectors.load(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\model.kv", mmap='r')

### Récupérer tous les mots du vocabulaire ###
vocab_words = w2v.index_to_key
print(len(vocab_words))

print(f" Vocabulaire chargé : {len(vocab_words)} mots")

### Sauvegarder dans un fichier texte  afin de voir les mots et cree le jeu de test ###
with open(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\vocabulaire_model.txt", "w", encoding="utf-8") as f:
    for word in vocab_words:
        f.write(word + "\n")

print(" Liste du vocabulaire enregistrée dans vocabulaire_model.txt")


