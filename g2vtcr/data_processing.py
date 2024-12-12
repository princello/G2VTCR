import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data,min_value_counts = 1000, max_value_counts = 20000):
    """Prepares and preprocesses raw input data for downstream analysis."""
    # Create a DataFrame with required columns
    imm = pd.DataFrame(np.array(raw_data).T, columns=['CDR3', 'Epitope', 'ids'])

    # Remove rows with asterisks in the 'CDR3' column
    asterisk_positions = imm['CDR3'].str.contains('\*', na=False)
    imm_raw = imm[~asterisk_positions]

    # Deduplicate based on 'sentence' (CDR3 + Epitope)
    imm_raw['sentence'] = imm_raw['CDR3'] + imm_raw['Epitope']
    imm_raw = imm_raw.drop_duplicates(subset='sentence')

    # Filter epitopes based on value counts
    epitope_counts = imm_raw['Epitope'].value_counts()
    selected_epitopes = epitope_counts[(epitope_counts > min_value_counts) & (epitope_counts < max_value_counts)].index

    imm_sel = imm_raw[imm_raw['Epitope'].isin(selected_epitopes)]

    return imm_raw, imm_sel 

def encode_seq(sequence, max_l):
    """Encodes a sequence into a numerical representation."""
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','*']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # Truncate sequence to max length
    sequence = sequence[:max_l]
    
    # Integer encode the sequence
    integer_encoded = [char_to_int[char] for char in sequence]
    
    # One-hot encode the sequence
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    # Pad the sequence to max length and flatten
    return np.pad(np.array(onehot_encoded), [(0, max_l - len(sequence)), (0, 0)], mode='constant').reshape(-1)


def train_test_split_imm(test_out_num, train_out_num, imm_raw, split_method="random", seed=None):
    """
    Splits data into train and test sets.

    Parameters:
    - test_out_num: int, maximum number of test samples per epitope.
    - train_out_num: int, maximum number of train samples per epitope.
    - imm_raw: DataFrame, processed data with unique CDR3 sequences and integer ids.
    - split_method: str, splitting method ("random", "similar").
    - seed: int, random seed for reproducibility.

    Returns:
    - imm_test: DataFrame, test set.
    - imm_train: DataFrame, train set.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    imm_train = pd.DataFrame(columns=imm_raw.columns)
    imm_test = pd.DataFrame(columns=imm_raw.columns)
    ids_list = list(set(imm_raw['ids']))

    for ids in ids_list:
        spe_CDR3 = imm_raw[imm_raw['ids'] == ids]
        CDR3_list = list(spe_CDR3['CDR3'].unique())
        total_num = len(CDR3_list)
        test_num = int(total_num * 0.2)

        if split_method == "random":
            neighbors = random.sample(CDR3_list, min(test_num, len(CDR3_list)))
        elif split_method == "similar":
            oh_emb = pd.Series(CDR3_list).apply(lambda x: encode_seq(x, 24))
            encoded_sequences = np.array(list(oh_emb))
            d = encoded_sequences.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(encoded_sequences)
            neighbors = random.sample(CDR3_list, test_num)

        for epitope in spe_CDR3['Epitope'].unique():
            epitope_df = spe_CDR3[spe_CDR3['Epitope'] == epitope]
            test_candidates = epitope_df[epitope_df['CDR3'].isin(neighbors)]
            train_candidates = epitope_df[~epitope_df['CDR3'].isin(neighbors)]

            test_df = test_candidates.sample(n=min(test_out_num, len(test_candidates)))
            train_df = train_candidates.sample(n=min(train_out_num, len(train_candidates)))

            imm_test = pd.concat((imm_test, test_df))
            imm_train = pd.concat((imm_train, train_df))

    return imm_test, imm_train
