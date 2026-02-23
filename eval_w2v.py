import numpy as np
import subprocess
import os
import time

base_dir = r"C:\Users\zinou\Downloads\M2\ML\TPML\2"
train_script = os.path.join(base_dir, "w2v.py")
eval_file = os.path.join(base_dir, "fichier evaluation pour le tp.txt")
result_file = os.path.join(base_dir, "resultats_final_piste1.txt")


def sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_embeddings(words_rep_path, eval_data_path):

    words_vects = {}
    with open(words_rep_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            words_vects[word] = vector


    erreurs = []
    N, accuracy = 0, 0

    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            mot, motpos, motneg = line.split()

            if mot not in words_vects or motpos not in words_vects or motneg not in words_vects:
                continue

            N += 1
            sim1 = sim(words_vects[mot], words_vects[motpos])
            sim2 = sim(words_vects[mot], words_vects[motneg])

            if sim1 > sim2:
                accuracy += 1
            else:
                erreurs.append((mot, motpos, motneg, sim1, sim2))


    errors_path = words_rep_path.replace(".txt", "_erreurs.txt")
    with open(errors_path, "w", encoding="utf-8") as ferr:
        ferr.write("mot motpos motneg sim(mot,motpos) sim(mot,motneg)\n")
        for mot, motpos, motneg, s1, s2 in erreurs:
            ferr.write(f"{mot} {motpos} {motneg} {s1:.4f} {s2:.4f}\n")

    print(f"  → Fichier des erreurs généré : {errors_path}")

    return accuracy / N if N > 0 else 0



scores = []
times = []

for i in range(1):
    print(f"\n Expérience {i+1}/10 : entraînement du modèle ")

    emb_path = os.path.join(base_dir, f"embeddings_run{i+1}_piste1.txt")

    start_time = time.time()

    # Entraîner le modèle
    subprocess.run(["python", train_script, emb_path], check=True)

    end_time = time.time()
    duration = end_time - start_time
    times.append(duration)

    # Évaluer les embeddings
    acc = evaluate_embeddings(emb_path, eval_file)
    scores.append(acc)

    print(f" Run {i+1}: accuracy = {acc:.4f} | temps = {duration:.2f} sec")




mean_score = np.mean(scores)
std_score = np.std(scores)
mean_time = np.mean(times)

print("\n")
print(f"Moyenne (10 runs)     : {mean_score:.4f}")
print(f"Écart-type (10 runs)  : {std_score:.4f}")
print(f"Temps moyen (s)       : {mean_time:.2f}")
print("\n")

with open(result_file, "w", encoding="utf-8") as f:
    f.write("Résultats de l’évaluation sur 10 entraînements indépendants\n")
    for i, (score, t) in enumerate(zip(scores, times)):
        f.write(f"Run {i+1:2d} : accuracy = {score:.4f} | temps = {t:.2f} sec\n")

    f.write(f"\nMoyenne accuracy : {mean_score:.4f}\n")
    f.write(f"Écart-type       : {std_score:.4f}\n")
    f.write(f"Temps moyen (s)  : {mean_time:.2f}\n")

print(f" Résultats sauvegardés dans : {result_file}")
