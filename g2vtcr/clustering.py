import faiss
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def cluster_tcrs(embeddings, eps=0.5, min_samples=5):
    """
    Performs DBSCAN clustering on TCR embeddings.

    Parameters:
    - embeddings: np.array, TCR embeddings to cluster.
    - eps: float, maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: int, number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - cluster_labels: np.array, cluster labels for each embedding.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

def visualize_tsne(embeddings, labels, ids, top_ids=None):
    """
    Visualizes TCR embeddings using t-SNE.

    Parameters:
    - embeddings: np.array, TCR embeddings to visualize.
    - labels: np.array, cluster labels for each embedding.
    - ids: list, epitope IDs associated with each embedding.
    - top_ids: list, IDs to highlight in the visualization. Others will be labeled as 'others'.

    Returns:
    - fig: plotly.graph_objects.Figure, t-SNE scatter plot.
    """
    from sklearn.manifold import TSNE
    import plotly.express as px

    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(embeddings)

    data = pd.DataFrame({
        't-SNE 1': tsne_components[:, 0],
        't-SNE 2': tsne_components[:, 1],
        'ids': ids,
        'labels': labels
    })

    if top_ids is not None:
        data['top_ids'] = data['ids'].copy()
        data.loc[~data['top_ids'].isin(top_ids), 'top_ids'] = 'others'
    else:
        data['top_ids'] = data['ids']

    custom_color_sequence = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10

    fig = px.scatter(
        data, x='t-SNE 1', y='t-SNE 2', color='top_ids',
        color_discrete_sequence=custom_color_sequence,
        labels={'t-SNE 1': 't-SNE 1', 't-SNE 2': 't-SNE 2'},
        title='t-SNE Visualization of TCR Clusters'
    )

    fig.update_layout(
        height=800, width=1200,
        margin=dict(l=50, r=50, b=50, t=50),
        plot_bgcolor='white'
    )

    return fig
