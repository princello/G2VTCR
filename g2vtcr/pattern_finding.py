import os
import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Added missing import
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def find_patterns(seq2vec, documents, cluster_labels, min_doc_freq=5, top_n=10):
    """
    Identifies significant patterns or motifs within clustered TCR sequences using TF-IDF.

    Parameters:
    - seq2vec: dict, mapping sequences to their embeddings (output of embed_seq).
    - documents: list, Weisfeiler-Lehman hashed documents (output of embed_seq).
    - cluster_labels: np.array, labels indicating the cluster for each sequence.
    - min_doc_freq: int, minimum document frequency for considering a feature.
    - top_n: int, number of top features to extract.

    Returns:
    - cluster_keywords: dict, top features for each cluster.
    - clustered_sequences: dict, mapping of clusters to their sequences.
    """

    # Convert seq2vec to DataFrame
    sequences = list(seq2vec.keys())
    embeddings = np.array(list(seq2vec.values()))
    data = pd.DataFrame(embeddings, index=sequences)
    data['cluster'] = cluster_labels

    # Group documents and sequences by cluster
    clustered_documents = {i: [] for i in np.unique(cluster_labels)}
    clustered_sequences = {i: [] for i in np.unique(cluster_labels)}

    for doc, label, seq in zip(documents, cluster_labels, sequences):
        clustered_documents[label].append(doc)
        clustered_sequences[label].append(seq)

    # Perform TF-IDF analysis
    vectorizer = TfidfVectorizer()
    all_docs = [doc for docs in clustered_documents.values() for doc in docs]
    X_all = vectorizer.fit_transform([" ".join(doc.get_graph_features()) for doc in all_docs])
    features = vectorizer.get_feature_names_out()

    def get_top_n_words(cluster_docs):
        X_cluster = vectorizer.transform([" ".join(doc.get_graph_features()) for doc in cluster_docs])
        tfidf_sum = np.asarray(X_cluster.sum(axis=0)).flatten()
        doc_freq = np.asarray((X_cluster > 0).sum(axis=0)).flatten()

        filtered_features = [
            (features[i], tfidf_sum[i])
            for i in range(len(features))
            if doc_freq[i] >= min_doc_freq
        ]
        filtered_features.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in filtered_features[:top_n]]

    # Extract top features for each cluster
    cluster_keywords = {
        cluster: get_top_n_words(docs)
        for cluster, docs in clustered_documents.items()
    }

    return cluster_keywords, clustered_sequences


def visualize_cluster_patterns(cluster_keywords, clustered_sequences, save_path="logo"): 
    """
    Visualizes patterns as sequence logos for each cluster.

    Parameters:
    - cluster_keywords: dict, mapping of clusters to their top features.
    - clustered_sequences: dict, mapping of clusters to their sequences.
    - save_path: str, directory path to save logos.
    """
    
    os.makedirs(save_path, exist_ok=True)

    res_len_dct = {'G':0, 'A':1, 'V':3, 'L':4, 'I':4, 'M': 4, 'P': 3, 'F': 7, 'W': 10,
              'S': 2, 'T': 3, 'Y': 8, 'N':4, 'C': 2, 'Q': 5, 'D': 4, 'E': 5, 'K':5,
              'R':7, 'H':6}

    def nodes_number(seq, pos_now):
        total_len = 0
        highlight_list = []
        for i, s in enumerate(seq):
            total_len += 4
            total_len += res_len_dct[s]
            highlight_list.extend(list(range(total_len-res_len_dct[s], total_len, 1)))
            if total_len > pos_now:
                break
        return seq, s, seq[max(0, i - 2):i + 3], highlight_list

    for cluster, features in cluster_keywords.items():
        for feature in features:
            if len(clustered_sequences[cluster]) > 20:
                logo_list = []
                for seq in clustered_sequences[cluster]:
                    seq_end, _, seq3, _ = nodes_number(seq, 0)  # Example highlighting
                    logo_list.append(seq3)

                amino_acids = sorted(list(set(''.join(logo_list))))
                position_counts = {i: {aa: 0 for aa in amino_acids} for i in range(-2, 3)}

                for seq in logo_list:
                    for i, aa in enumerate(seq):
                        position = i - 2
                        position_counts[position][aa] += 1

                total_sequences = len(logo_list)
                position_freqs = {
                    pos: {aa: count / total_sequences for aa, count in counts.items()}
                    for pos, counts in position_counts.items()
                }

                aa_df = pd.DataFrame(position_freqs).T

                fig, ax = plt.subplots(figsize=(3, 1))
                ss_logo = logomaker.Logo(aa_df,
                                         ax=ax,
                                         width=.8,
                                         vpad=.05,
                                         fade_probabilities=True,
                                         stack_order='small_on_top',
                                         color_scheme='classic')

                ss_logo.style_spines(spines=['left', 'right'], visible=False)
                ss_logo.ax.set_xticks([-2, -1, 0, 1, 2])
                ss_logo.ax.set_xticklabels('%+d'%x for x in [-2, -1, 0, 1, 2])
                ss_logo.ax.set_ylabel('Probability')

                plt.savefig(f'{save_path}/logo_cluster_{cluster}_{feature}.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig)