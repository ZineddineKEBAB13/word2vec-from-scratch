from gensim.models import KeyedVectors
from scipy.spatial import KDTree
import numpy as np
import time

def tester_analogies_kdtree(fichier, model, tree, mots, output_file):
    start_time = time.time()

    total, correct = 0, 0
    bons = []

    cat_name = None
    stats_cat = {} 

    with open(fichier, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()

            if ligne.startswith("#"):
                cat_name = ligne.replace("#", "").strip()
                stats_cat[cat_name] = {"total": 0, "correct": 0}
                continue

            if not ligne:
                continue

            w = ligne.split()
            if len(w) != 4:
                continue

            a, b, c, d = w
            if any(x not in model.key_to_index for x in (a,b,c,d)):
                continue


            vec = model[b] - model[a] + model[c]
            dist, idx = tree.query(vec, k=1)
            mot_pred = mots[idx]

            total += 1
            stats_cat[cat_name]["total"] += 1

            if mot_pred == d:
                correct += 1
                stats_cat[cat_name]["correct"] += 1
                bons.append((a, b, c, d))
    
    end_time = time.time()
    duration = end_time - start_time
    taux_global = 100 * correct / total if total > 0 else 0

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"FICHIER : {fichier}\n")
        out.write(f"TOTAL : {correct}/{total} corrects ({taux_global:.2f}%)\n")
        out.write(f"Temps total : {duration:.3f} sec\n\n")

        out.write("=== Accuracy par catégorie ===\n")
        for cat, st in stats_cat.items():
            if st["total"] == 0:
                acc = 0
            else:
                acc = 100 * st["correct"] / st["total"]
            out.write(f"- {cat} : {st['correct']}/{st['total']}  ({acc:.2f}%)\n")
        out.write("\n")

        out.write("=== Quadruplets bien prédits ===\n")
        for a, b, c, d in bons:
            out.write(f"{a} : {b} :: {c} : {d}\n")

    print(f"\nRésultats écrits dans : {output_file}")


w2v = KeyedVectors.load(r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\model.kv", mmap='r')

mots = list(w2v.key_to_index.keys())
vecteurs = np.array([w2v[m] for m in mots])
tree = KDTree(vecteurs)


tester_analogies_kdtree(
    r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\semantique.txt",
    w2v, tree, mots,
    output_file="resultats_semantique.txt"
)

tester_analogies_kdtree(
    r"C:\Users\zinou\Downloads\M2\ML\TPML\2\data\syntaxique.txt",
    w2v, tree, mots,
    output_file="resultats_syntaxique.txt"
)
