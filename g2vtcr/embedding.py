import networkx as nx
from rdkit import Chem
from karateclub import Graph2Vec
from torch_geometric.utils import from_networkx, from_smiles
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

import networkx as nx
from rdkit import Chem
from karateclub import Graph2Vec
from torch_geometric.utils import from_networkx, from_smiles
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing

def embed_seq(imm_raw, wl_iterations=5, dimensions=256, workers=4, down_sampling=0.0001, epochs=10, learning_rate=0.025, min_count=5, seed=42, attributed=False):
    """
    Generates graph-based embeddings for TCR and epitope sequences.

    Parameters:
    - imm_raw: DataFrame, processed data with unique CDR3 and Epitope sequences.
    - wl_iterations: int, number of Weisfeiler-Lehman iterations for hashing.
    - dimensions: int, size of the embedding vectors.
    - workers: int, number of worker threads for training.
    - down_sampling: float, threshold for down-sampling.
    - epochs: int, number of training epochs.
    - learning_rate: float, learning rate for training.
    - min_count: int, minimum count for vocabulary trimming.
    - seed: int, random seed for reproducibility.

    Returns:
    - seq2vec: dict, mapping of sequences to their embeddings.
    """

    # Extract unique CDR3 and Epitope sequences
    trb_set = imm_raw.drop_duplicates('CDR3')
    epi_set = imm_raw.drop_duplicates('Epitope')

    # Convert sequences to RDKit Mol objects
    trb_mol = trb_set['CDR3'].apply(lambda x: Chem.MolFromFASTA(x))
    epi_mol = epi_set['Epitope'].apply(lambda x: Chem.MolFromFASTA(x))

    # Combine TCR and Epitope molecules
    seq = pd.concat([trb_set['CDR3'], epi_set['Epitope']], axis=0, ignore_index=True)
    mol = pd.concat([trb_mol, epi_mol], axis=0, ignore_index=True)

    # Convert molecules to graphs
    graph = mol.apply(lambda x: mol_to_nx(x))
    seq2graph = dict(zip(seq, graph))

    # Prepare documents for embedding
    documents = [WeisfeilerLehmanHashing(g, wl_iterations, erase_base_features=False, attributed=attributed) for g in graph]
    tagged_documents = [TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(documents)]

    # Train Doc2Vec model
    doc2vec_model = Doc2Vec(
        tagged_documents,
        vector_size=dimensions,
        window=0,
        min_count=min_count,
        dm=0,
        sample=down_sampling,
        workers=workers,
        epochs=epochs,
        alpha=learning_rate,
        seed=seed,
    )

    # Generate embeddings
    embeddings = [doc2vec_model.dv[str(i)] for i, _ in enumerate(tagged_documents)]
    seq2vec = dict(zip(seq, embeddings))

    return seq2vec

def mol_to_nx(mol, node_features=None, edge_features=None):
    """
    Converts a molecule to a NetworkX graph with customizable node and edge features.

    Parameters:
    - mol: RDKit molecule object.
    - node_features: dict, optional. Defines the attributes to extract for nodes.
      Default includes 'atomic_num', 'is_aromatic', and 'atom_symbol'.
    - edge_features: dict, optional. Defines the attributes to extract for edges.
      Default includes 'bond_type'.

    Returns:
    - G: NetworkX graph with nodes and edges annotated with the specified features.
    """
    # Default node and edge feature definitions
    if node_features is None:
        node_features = {
            'atomic_num': lambda atom: atom.GetAtomicNum(),
            'is_aromatic': lambda atom: atom.GetIsAromatic(),
            'atom_symbol': lambda atom: atom.GetSymbol()
        }

    if edge_features is None:
        edge_features = {
            'bond_type': lambda bond: bond.GetBondType()
        }

    G = nx.Graph()

    # Add nodes with custom features
    for atom in mol.GetAtoms():
        node_attributes = {key: func(atom) for key, func in node_features.items()}
        G.add_node(atom.GetIdx(), **node_attributes)

    # Add edges with custom features
    for bond in mol.GetBonds():
        edge_attributes = {key: func(bond) for key, func in edge_features.items()}
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **edge_attributes)

    return G
