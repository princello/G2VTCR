"""
G2VTCR - Graph-based TCR sequence analysis package
=================================================

This package provides functions for analyzing T-cell receptor (TCR) sequences 
using graph-based embedding approaches.

Modules:
--------
data_processing: Data preparation and preprocessing
embedding: Graph-based sequence embedding
clustering: TCR sequence clustering
pattern_finding: Identifying motifs in TCR clusters
classification: TCR-epitope interaction prediction

"""

# Import main functions for easy access
from .data_processing import preprocess_data, train_test_split_imm, encode_seq
from .embedding import embed_seq, mol_to_nx
from .clustering import cluster_tcrs, visualize_tsne
from .pattern_finding import find_patterns, visualize_cluster_patterns
from .classification import generate, predict_interactions

__version__ = '0.1.0'