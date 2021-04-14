from typing import List
from pathlib import Path
import warnings
import logging
import pickle
import glob

from natsort import natsorted

import numpy as np
import pandas as pd
import plotly.express as px

import embviz.utils as utils


def load_reducer(method: str, n_components: int):
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components,
                       n_iter=500, verbose=1, n_jobs=5, random_state=0)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')

    return reducer


class EmbeddingSpaceLogger:

    def __init__(self, root: str, n_components: int = 3, method='umap'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.n_components = n_components
        self.method = method
        self.reducer = load_reducer(method, n_components)

    def keys(self):
        keys = glob.glob(str(self.root / '*.emb'))
        keys = [Path(k).stem for k in keys]
        keys = natsorted(list(keys))
        return keys

    def set_method(self, method: str):
        self.reducer = load_reducer(method, self.n_components)
        self.method = method

    def set_n_components(self, n_components: int):
        self.reducer = load_reducer(self.method, n_components)
        self.n_components = n_components

    def get_step(self, step_key: str):
        with open(self.root / f'{step_key}.emb', 'rb') as f:
            return pickle.load(f)

    def add_step(self, step_key: str, embeddings: List[np.ndarray], labels: List[str],
                 symbols: List[str] = None, metadata: List[dict] = None,
                 **plotly_kwargs):
        step_key = utils.safe_string(step_key)
        if (self.root / f'{step_key}.emb').exists():
            warnings.warn(f'step_key  {step_key} already found in logs. \
                This will overwrite any previous data')

        step = {
            'embedding': embeddings,
            'labels': labels,
            'symbols': symbols,
            'metadata': metadata,
            'plotly_kwargs': plotly_kwargs,
            'projs': {'2d': {},  '3d': {}},
        }

        with open(self.root / f'{step_key}.emb', 'wb') as f:
            pickle.dump(step, f)

    def update_step(self, step_key: str, step: dict):
        # TODO: wrappers would be a more elegant solution to this I think
        with open(self.root / f'{step_key}.emb', 'wb') as f:
            pickle.dump(step, f)

    def plot_step(self, key):
        step = self.get_step(key)

        symbols = step['symbols']
        labels = step['labels']
        metadata = step['metadata']

        if self.method in step['projs'][f'{self.n_components}d']:
            projection = step['projs'][f'{self.n_components}d'][self.method]
        else:
            print(
                f'doing {self.method} {self.n_components} dim reduction for {key}')
            projection = self.reducer.fit_transform(step['embedding'])

            print(f'caching...')
            step['projs'][f'{self.n_components}d'][self.method] = projection
            self.update_step(key, step)

            print(f'dim reduction done')

        scatter_fn = px.scatter if self.n_components == 2 else px.scatter_3d
        axes = ('x', 'y', 'z')
        proj = {}
        scatter_kwargs = {}
        for idx in range(projection.shape[-1]):
            proj[axes[idx]] = projection[:, idx]
            scatter_kwargs[axes[idx]] = axes[idx]

        df = pd.DataFrame(dict(
            label=labels,
            audio_path=metadata['audio_path'],
            **proj,
        ))
        fig = scatter_fn(df, color='label',
                         symbol=symbols,
                         custom_data=['audio_path', 'label'],
                         color_discrete_sequence=px.colors.qualitative.Light24,
                         **scatter_kwargs,
                         **step['plotly_kwargs'])

        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        return fig
