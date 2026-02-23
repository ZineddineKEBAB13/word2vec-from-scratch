from collections import Counter
import numpy as np
import random


def sigmoid(x: np.ndarray) :
    return 1.0 / (1.0 + np.exp(-x))


def read_corpus(path):
    
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            #if len(tokens) > 4:
            #    tokens = tokens[2:-2]
            #else:
            #    tokens = []
            
            if tokens:
                sentences.append(tokens)
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
    filtered_sentences = []
    for sent in sentences:
        kept = [w for w in sent if w in word2id]
        if kept:
            filtered_sentences.append(kept)
    return filtered_sentences


def compute_negative_distribution(unigram_counts, word2id, alpha=0.75):
    V = len(word2id)
    freqs = np.zeros(V, dtype=np.float64)
    for w, idx in word2id.items():
        freqs[idx] = unigram_counts[w]
    freqs_pow = freqs ** alpha
    probs = freqs_pow / freqs_pow.sum()
    return probs 


def build_training_pairs(sentences, word2id, window_size):
    
    pairs = []
    for sent in sentences:
        ids = [word2id[w] for w in sent]
        n = len(ids)
        for i, center in enumerate(ids):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, ids[j]))
    return pairs


def init_embeddings(vocab_size, dim):
    M = np.random.uniform(-0.5/dim, 0.5/dim, size=(vocab_size, dim)).astype(np.float64)
    C = np.random.uniform(-0.5/dim, 0.5/dim, size=(vocab_size, dim)).astype(np.float64)
    return M, C


def gradient_update(eta, m_vec, cpos_vec, cneg_vecs):
    
    score_pos = np.dot(m_vec, cpos_vec)
    
    s_pos = sigmoid(score_pos)

    scores_neg = cneg_vecs @ m_vec
    s_negs = sigmoid(scores_neg)

    grad_cpos = (s_pos - 1.0) * m_vec
    grad_cnegs = (s_negs[:, np.newaxis] * m_vec[np.newaxis, :])
    grad_m = (s_pos - 1.0) * cpos_vec + np.sum(s_negs[:, np.newaxis] * cneg_vecs, axis=0)
    
    m_vec_new = m_vec - eta * grad_m
    cpos_vec_new = cpos_vec - eta * grad_cpos
    cneg_vecs_new = cneg_vecs - eta * grad_cnegs

    return m_vec_new, cpos_vec_new, cneg_vecs_new


def save_embeddings(M, id2word, filename):
    nb_mots, dim = M.shape
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{nb_mots} {dim}\n")
        for idx, mot in enumerate(id2word):
            vect_str = " ".join(str(x) for x in M[idx])
            f.write(f"{mot} {vect_str}\n")



def w2v(corpus_path,eta,dim,k,window_size,epochs=5,minc=5,alpha=0.75,output_path="C:\\Users\\zinou\\Downloads\\M2\\ML\\TPML\\2\\embeddings.txt"
):
    
    sentences = read_corpus(corpus_path)

    unigram_counts = compute_unigram_counts(sentences)

    vocab, word2id, id2word = build_vocab(unigram_counts, minc)

    sentences = filter_sentences(sentences, word2id)

    unigram_counts = compute_unigram_counts(sentences)

    neg_probs = compute_negative_distribution(unigram_counts, word2id, alpha)
    
    all_ids = np.arange(len(vocab))

    training_pairs = build_training_pairs(sentences, word2id, window_size)

    vocab_size = len(vocab)
    
    M, C = init_embeddings(vocab_size, dim)

    for epoch in range(epochs):
        random.shuffle(training_pairs)
        print(f" Époque {epoch+1}/{epochs}")

        for center_id, pos_id in training_pairs:
            
            m_vec = M[center_id]
            
            cpos_vec = C[pos_id]

            neg_ids = np.random.choice(all_ids, size=k, p=neg_probs, replace=True)
            
            cneg_vecs = C[neg_ids]

            m_vec_new, cpos_vec_new, cneg_vecs_new = gradient_update(eta, m_vec, cpos_vec, cneg_vecs)

            M[center_id] = m_vec_new
            C[pos_id] = cpos_vec_new
            C[neg_ids] = cneg_vecs_new


    save_embeddings(M, id2word, output_path)
    print(f" Entraînement terminé. Embeddings sauvegardés dans : {output_path}")

    return M, C, word2id, id2word


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = r"C:\Users\zinou\Downloads\M2\ML\TPML\2\embeddings.txt"

    corpus = r"C:\Users\zinou\Downloads\M2\ML\TPML\1 final\tlnl_tp1_data\alexandre_dumas\Le_comte_de_Monte_Cristo.tok"

    print(f" Entraînement du modèle Word2Vec — sortie : {output_path}")
    w2v(corpus,eta=0.1,dim=100,k=10, window_size=2, epochs=5, minc=5, alpha=1, output_path=output_path)
