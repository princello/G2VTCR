# G2VTCR
Graph-based Representation and Embedding of Antigen and TCR for Enhanced Recognition Analysis
![G2VTCR Workflow](images/Fig1.png "Graph-based Workflow of G2VTCR")
*Figure 1. G2VTCR framework showing graph-based TCR and epitope representation.*

## Description
### Purpose
T cells are essential mediators of the adaptive immune response, playing a critical role in detecting infections and facilitating vaccination efficacy through antigen recognition via T cell receptors (TCRs). The **G2VTCR** framework empowers researchers by providing predictive insights into TCR-epitope specificity and facilitating TCR clustering. This tool addresses the vast and largely unexplored space of TCR-antigen interactions, which cannot be fully captured by experimental methods alone.

### Principle
The G2VTCR framework represents TCR and epitope sequences as atomic-level graphs, leveraging the structural and contextual information inherent in these sequences. By embedding TCRs and epitopes into a graph-based representation, the model predicts the interaction probability between a given TCR and epitope. A value close to one indicates a high likelihood of recognition by the TCR.

### Technical Details
G2VTCR employs a graph-based approach and a robust two-step training pipeline to capture biologically meaningful interactions:

- Graph Representation and Embedding:
TCR and epitope sequences are transformed into atomic-level graphs, where atoms represent nodes and chemical bonds represent edges. Weisfeiler-Lehman (WL) iterations are used to capture rich contextual subgraphs within these representations, enhancing the model's ability to interpret molecular interactions.

- G2VTCR leverages graph-based representations to perform either unsupervised clustering or supervised classification, depending on the research goal:
  - Clustering (Unsupervised)
TCR sequences are embedded into an atomic-level graph-based representation, capturing structural and contextual subgraph features using Weisfeiler-Lehman (WL) iterations. These embeddings are then clustered to group TCRs with similar structural and functional properties. This step highlights inherent relationships within the TCR repertoire, providing insights into their diversity and functional groupings.
  - Classification (Supervised)
Pairwise embeddings of TCR and epitope graphs are utilized to train a classification model. This step predicts the interaction probability between a given TCR and epitope pair, with the model output indicating the likelihood of recognition. The classification task ensures high specificity in identifying meaningful TCR-epitope bindings.

## Installation

First, make sure you have the required dependencies:

```bash
pip install numpy pandas sklearn rdkit-pypi networkx faiss-cpu karateclub gensim torch-geometric logomaker matplotlib 
```

You can also install the package directly:

```bash
pip install git+https://github.com/princello/G2VTCR.git
```

## Quick Tutorial

### Data Processing
Prepare and preprocess the raw input data for downstream analysis.

```python
# Import required functions
from g2vtcr import preprocess_data
import pandas as pd

# Prepare raw data (example format)
raw_data = [
    ['CASSLAPGATNEKLFF', 'ASSLPTTMNY', 0],  # [CDR3, Epitope, ids]
    ['CASSLSFGTEAFF', 'FRDYVDRFYKTLRAEQASQE', 1],
    # ... more data
]

# Preprocess data
imm_raw, imm_sel = preprocess_data(
    raw_data,
    min_value_counts=100,  # Minimum samples per epitope
    max_value_counts=20000  # Maximum samples per epitope
)
```

### Graph-based Embedding
Generate embeddings for TCR and epitope sequences using graph representations.

```python
# Import embedding function
from g2vtcr import embed_seq
import numpy as np

# Generate sequence embeddings
seq2vec = embed_seq(
    imm_raw,
    wl_iterations=5,
    dimensions=256,
    workers=4,
    epochs=10
)

# Extract sequences and embeddings
sequences = list(seq2vec.keys())
embeddings = np.array(list(seq2vec.values()))
```

### Clustering
Perform DBSCAN clustering on TCR embeddings to identify groups with similar properties.

```python
# Import clustering functions
from g2vtcr import cluster_tcrs, visualize_tsne

# Perform clustering
cluster_labels = cluster_tcrs(
    embeddings,
    eps=0.5,
    min_samples=5
)

# Optional: Visualize clustering with t-SNE
ids = imm_raw['ids'].values[:len(embeddings)]
fig = visualize_tsne(embeddings, cluster_labels, ids)
# fig.show()  # Display the visualization
```

### Pattern Finding
Identify significant patterns or motifs within the clustered TCR sequences.

```python
# Import pattern finding functions
from g2vtcr import find_patterns, visualize_cluster_patterns, mol_to_nx
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from rdkit import Chem

# Prepare documents for pattern finding
documents = []
for seq in sequences:
    mol = Chem.MolFromFASTA(seq)
    graph = mol_to_nx(mol)
    doc = WeisfeilerLehmanHashing(graph, wl_iterations=5, erase_base_features=False)
    documents.append(doc)

# Find patterns
cluster_keywords, clustered_sequences = find_patterns(
    seq2vec,
    documents,
    cluster_labels,
    min_doc_freq=5,
    top_n=10
)

# Visualize patterns as sequence logos
visualize_cluster_patterns(
    cluster_keywords,
    clustered_sequences,
    save_path="logo"
)
```

### Classification
Predict TCR-epitope interactions using a supervised learning approach.

```python
# Import classification functions
from g2vtcr import generate, predict_interactions, train_test_split_imm

# First, split data into train and test sets
imm_test, imm_train = train_test_split_imm(
    test_out_num=10,    # Max test samples per epitope
    train_out_num=40,   # Max train samples per epitope
    imm_raw=imm_sel,
    split_method="random",
    seed=42
)

# Generate positive and negative samples for classification
train_samples = generate(
    train=imm_train,
    nn=1,               # Ratio of negative to positive samples
    name="train",       # Base name for output files
    min_value_counts=100
)

test_samples = generate(
    train=imm_test,
    nn=1,
    name="test",
    min_value_counts=100
)

# Predict TCR-epitope interactions
auc_score = predict_interactions(
    seq2vec=seq2vec,
    train_name="train",
    test_name="test",
    random_state=42
)

print(f"AUC Score: {auc_score}")
```

## Complete Example Pipeline

Here's a complete example showing the full pipeline:

```python
# Import all required modules
from g2vtcr import (
    preprocess_data, embed_seq, cluster_tcrs,
    find_patterns, visualize_cluster_patterns,
    train_test_split_imm, generate, predict_interactions,
    mol_to_nx
)
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from rdkit import Chem
import numpy as np
import pandas as pd

# Step 1: Prepare raw data
raw_data = [
    ['CASSLAPGATNEKLFF', 'GILGFVFTL', 0],    # [CDR3, Epitope, epitope_id]
    ['CASSLGQGYEQYF', 'GILGFVFTL', 0],
    ['CASSLSFGTEAFF', 'NLVPMVATV', 1],
    # ... more data
]

# Step 2: Preprocess data
imm_raw, imm_sel = preprocess_data(raw_data, min_value_counts=5, max_value_counts=500)

# Step 3: Generate embeddings
seq2vec = embed_seq(imm_raw)

# Extract sequences and embeddings
sequences = list(seq2vec.keys())
embeddings = np.array(list(seq2vec.values()))

# Step 4: Perform clustering
cluster_labels = cluster_tcrs(embeddings, eps=0.5, min_samples=2)

# Step 5: Find patterns
# Prepare documents for pattern finding
documents = []
for seq in sequences:
    mol = Chem.MolFromFASTA(seq)
    graph = mol_to_nx(mol)
    doc = WeisfeilerLehmanHashing(graph, wl_iterations=5)
    documents.append(doc)

# Extract patterns
cluster_keywords, clustered_sequences = find_patterns(
    seq2vec, documents, cluster_labels, min_doc_freq=2, top_n=5
)

# Visualize patterns
visualize_cluster_patterns(cluster_keywords, clustered_sequences, save_path="logos")

# Step 6: Split data for classification
imm_test, imm_train = train_test_split_imm(5, 10, imm_sel, "random", 42)

# Step 7: Generate training and test samples
train_samples = generate(imm_train, 1, "train", 100)
test_samples = generate(imm_test, 1, "test", 100)

# Step 8: Predict interactions
auc_score = predict_interactions(seq2vec, "train", "test", 42)

print(f"Final AUC Score: {auc_score}")
```

## Advanced Usage

For more advanced usage and customization options, please refer to the documentation and examples in the source code.

## Contact and Support

Please report any problems directly to the GitHub [issue tracker](https://github.com/princello/G2VTCR/issues)

Also, you can send your feedback to zw2595@cumc.columbia.edu


