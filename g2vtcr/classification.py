import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def generate(train, nn, name, min_value_counts):
    """
    Generates data for training or testing with shuffled pairs as negative samples.

    Parameters:
    - train: DataFrame, input data with `CDR3`, `Epitope`, and `ids` columns.
    - nn: int, number of negative samples per positive sample.
    - name: str, base name for output files.
    - min_value_counts: int, minimum number of samples per epitope.

    Returns:
    - output: DataFrame, combined positive and negative samples with labels.
    """
    train['sentence'] = train['Epitope'] + train['CDR3']

    # Oversample positive samples to ensure sufficient representation
    train_oversampled = pd.concat([train] * nn, ignore_index=True)

    # Generate negative samples by shuffling
    shuffled_train = train_oversampled.sample(frac=1, random_state=1).reset_index(drop=True)
    negative_samples = pd.DataFrame({
        'CDR3': shuffled_train['CDR3'],
        'Epitope': train_oversampled['Epitope'],
        'label': 0
    })

    # Ensure no duplicates with positive samples
    negative_samples['sentence'] = negative_samples['Epitope'] + negative_samples['CDR3']
    negative_samples = negative_samples[~negative_samples['sentence'].isin(train['sentence'])]

    # Downsample positive and negative samples
    positive_samples = train.sample(n=min(len(train), min_value_counts), random_state=1)
    positive_samples['label'] = 1

    combined_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Save to files
    combined_samples[['CDR3']].to_csv(f'{name}_cdr3.tsv', sep='\t', index=False, header=False)
    combined_samples[['Epitope']].to_csv(f'{name}_epi.tsv', sep='\t', index=False, header=False)
    combined_samples[['label']].to_csv(f'{name}_label.tsv', sep='\t', index=False, header=False)

    return combined_samples

def predict_interactions(seq2vec, train_name, test_name, random_state=42):
    """
    Predicts TCR-epitope interactions using a supervised learning approach.

    Parameters:
    - seq2vec: dict, mapping sequences to embeddings (output of embed_seq).
    - train_name: str, base name for training files (e.g., 'vdj_train').
    - test_name: str, base name for testing files (e.g., 'vdj_test').
    - random_state: int, random seed for model reproducibility.

    Returns:
    - auc_score: float, area under the ROC curve for the predictions.
    """
    # Load training data
    trb_train_seq = pd.read_csv(f'{train_name}_cdr3.tsv', header=None)
    epi_train_seq = pd.read_csv(f'{train_name}_epi.tsv', header=None)
    y_train = pd.read_csv(f'{train_name}_label.tsv', header=None)[0].values.astype('float')

    # Prepare training embeddings
    trb_g2v_train_ss = trb_train_seq[0].apply(lambda x: seq2vec[x])
    epi_g2v_train_ss = epi_train_seq[0].apply(lambda x: seq2vec[x])

    X_train = np.concatenate((list(trb_g2v_train_ss), list(epi_g2v_train_ss)), axis=1)

    # Train classifier
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # Load test data
    trb_test_seq = pd.read_csv(f'{test_name}_cdr3.tsv', header=None)
    epi_test_seq = pd.read_csv(f'{test_name}_epi.tsv', header=None)
    y_test = pd.read_csv(f'{test_name}_label.tsv', header=None)[0].values.astype('float')

    # Prepare test embeddings
    trb_g2v_test_ss = trb_test_seq[0].apply(lambda x: seq2vec[x])
    epi_g2v_test_ss = epi_test_seq[0].apply(lambda x: seq2vec[x])

    X_test = np.concatenate((list(trb_g2v_test_ss), list(epi_g2v_test_ss)), axis=1)

    # Predict probabilities
    y_prob = clf.predict_proba(X_test)

    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    auc_score = auc(fpr, tpr)

    return auc_score
