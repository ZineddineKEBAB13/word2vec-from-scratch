from collections import Counter
import numpy as np
import random


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def read_corpus(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) > 4:
                toks = toks[2:-2]
                if toks:
                    sentences.append(toks)
    return sentences


def compute_unigram_counts(sentences):
    unigrams = Counter()
    for sent in sentences:
        unigrams.update(sent)
    return unigrams


def build_vocab(unigram_counts, min_count):
    vocab = [w for w, c in unigram_counts.items() if c >= min_count]
    vocab.sort()
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = vocab[:]
    return vocab, word2id, id2word


def filter_sentences(sentences, word2id):
    filtered = []
    for sent in sentences:
        kept = [w for w in sent if w in word2id]
        if kept:
            filtered.append(kept)
    return filtered


def build_negative_table(unigram_counts, word2id, alpha=0.75, table_size=1_000_000):
    vocab_size = len(word2id)
    freqs = np.zeros(vocab_size, dtype=np.float64)
    
    for w, idx in word2id.items():
        freqs[idx] = unigram_counts[w]
        
    train_words_pow = np.sum(freqs ** alpha)
    table = np.zeros(table_size, dtype=np.int32)
    
    i = 0
    d1 = (freqs[i] ** alpha) / train_words_pow
    for a in range(table_size):
        table[a] = i
        if (a / table_size) > d1:
            i += 1
            if i >= vocab_size:
              i = vocab_size - 1
            d1 += (freqs[i] ** alpha) / train_words_pow
            
            

    return table


def compute_keep_probs(unigram_counts, word2id, sample_t=1e-3):
    total = sum(unigram_counts[w] for w in word2id)
    keep_probs = np.zeros(len(word2id), dtype=np.float64)
    for w, idx in word2id.items():
        f = unigram_counts[w] / total
        p_keep = (np.sqrt(f / sample_t) + 1.0) * (sample_t / f)
        if p_keep > 1.0:
            p_keep = 1.0
        keep_probs[idx] = p_keep
    return keep_probs


def init_embeddings(vocab_size, dim):
    M = np.random.uniform(-0.5 / dim, 0.5 / dim, size=(vocab_size, dim)).astype(np.float64)
    C = np.random.uniform(-0.5 / dim, 0.5 / dim, size=(vocab_size, dim)).astype(np.float64)
    return M, C


def gradient_update(m_vec, cpos_vec, cneg_vecs):
    score_pos = np.dot(m_vec, cpos_vec)
    s_pos = sigmoid(score_pos)
    scores_neg = cneg_vecs @ m_vec
    s_negs = sigmoid(scores_neg)
    grad_cpos = (s_pos - 1.0) * m_vec
    grad_cnegs = (s_negs[:, np.newaxis] * m_vec[np.newaxis, :])
    grad_m = (s_pos - 1.0) * cpos_vec + np.sum(s_negs[:, np.newaxis] * cneg_vecs, axis=0)
    return grad_m, grad_cpos, grad_cnegs


def save_embeddings(M, id2word, filename):
    nb_mots, dim = M.shape
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{nb_mots} {dim}\n")
        for idx, mot in enumerate(id2word):
            vect_str = " ".join(str(x) for x in M[idx])
            f.write(f"{mot} {vect_str}\n")


def w2v(corpus_path, eta, dim, k, window_size, epochs=5,minc=5, alpha=0.75, sample_t=1e-3,output_path="embeddings.txt"):

    sentences = read_corpus(corpus_path)
    
    unigram_counts = compute_unigram_counts(sentences)
    vocab, word2id, id2word = build_vocab(unigram_counts, minc)
    sentences = filter_sentences(sentences, word2id)
    unigram_counts = compute_unigram_counts(sentences)

    neg_table = build_negative_table(unigram_counts, word2id, alpha=alpha)
    keep_probs = compute_keep_probs(unigram_counts, word2id, sample_t)

    vocab_size = len(vocab)
    M, C = init_embeddings(vocab_size, dim)

    
    word_counter_total = 0
    decay_step = 500 # 2.5 % du corpus 
    min_eta = eta * 0.0001
    total_train_words = sum(len(s) for s in sentences)
    eta_t = eta

    for epoch in range(epochs):
        
        word_counter = 0
        last_word_counter = 0 
        
        print(f"  Époque {epoch+1}/{epochs}")
        
        random.shuffle(sentences)
        for sent in sentences:
            ids = []
            
            for w in sent:
                word_counter += 1
                if random.random() <= keep_probs[word2id[w]]:
                    ids.append(word2id[w])
                

            n = len(ids)
            
            for i, center_id in enumerate(ids):
                
                #  mise à jour du learning rate
                if i > 0 and (word_counter - last_word_counter >=  decay_step) :                    
                    word_counter_total += word_counter - last_word_counter
                    last_word_counter = word_counter
                    progress = word_counter_total / ((epochs * total_train_words) + 1)
                    eta_t = max(eta * (1.0 - progress), min_eta)
                    
                
                start = max(0, i - window_size)
                end = min(n, i + window_size + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    pos_id = ids[j]

                    # Tirage négatif dans la table
                    rand_idx = np.random.randint(0, neg_table.shape[0], size=k)
                    neg_ids = neg_table[rand_idx]
                    
                    for z in range(k):
                        while neg_ids[z] == center_id:
                            neg_ids[z] = neg_table[np.random.randint(0, neg_table.shape[0])]

                    m_vec = M[center_id]
                    cpos_vec = C[pos_id]
                    cneg_vecs = C[neg_ids]

                    grad_m, grad_cpos, grad_cnegs = gradient_update(m_vec, cpos_vec, cneg_vecs)

                    M[center_id] -= eta_t * grad_m
                    C[pos_id] -= eta_t * grad_cpos
                    for idx, neg_id in enumerate(neg_ids):
                        C[neg_id] -= eta_t * grad_cnegs[idx]
                    
                    
        word_counter_total += word_counter - last_word_counter     
        word_counter = 0
        last_word_counter = 0
        
        
    save_embeddings(M, id2word, output_path)
    print(f"\n Entraînement terminé. Embeddings sauvegardés dans : {output_path}")
    return M, C, word2id, id2word


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else :
        output_path = r"C:\Users\zinou\Downloads\M2\ML\TPML\2\embeddings_ameliore.txt"
        
    corpus = r"C:\Users\zinou\Downloads\M2\ML\TPML\1 final\tlnl_tp1_data\alexandre_dumas\Le_comte_de_Monte_Cristo.tok"
    

    print("  Entraînement du modèle Word2Vec (skip-gram + table d'echantillionage + subsampling + fenetre aleatoire + decrementation de learning rate)...")
    w2v(corpus, eta=0.1, dim=100, k=10, window_size=2,
        epochs=5, minc=5, alpha=0.75, sample_t=1e-3,
        output_path=output_path)

