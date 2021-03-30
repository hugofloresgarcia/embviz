from typing import List
from pathlib import Path
from collections import OrderedDict
import warnings
import shelve

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
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')

    return reducer

class EmbeddingSpaceLogger:

    def __init__(self, path: str, n_components: int = 3, method='umap'):
        self.shelf_path = Path(path)

        self.n_components = n_components
        self.method = method
        self.reducer = load_reducer(method, n_components)

    def keys(self):
        with shelve.open(str(self.shelf_path), writeback=False) as shelf:
            keys = list(shelf.keys())
        return keys

    def get_step(self, step_key: str):
        with shelve.open(str(self.shelf_path), writeback=False) as shelf:
            if step_key in shelf:
                return dict(shelf[step_key])
            else:
                raise ValueError(f'{step_key}')

    def add_step(self, step_key: int, embeddings: List[np.ndarray], labels: List[str],
                 symbols: List[str] = None, metadata: List[dict] = None,
                 **plotly_kwargs):
        with shelve.open(str(self.shelf_path), writeback=True) as shelf:
            step_key = utils.safe_string(step_key)
            if step_key in shelf:
                warnings.warn(f'step_key  {step_key} already found in logs. \
                    This will overwrite any previous data')
        
            shelf[step_key] = {
                'embedding': embeddings,
                'labels': labels,
                'symbols': symbols,
                'metadata': metadata,
                'plotly_kwargs': plotly_kwargs
            }

            shelf.sync()

    def update_step(self, step_key: str, step: dict):
        #TODO: wrappers would be a more elegant solution to this I think
        with shelve.open(str(self.shelf_path), writeback=False) as shelf:
            if step_key in shelf:
                shelf[step_key] = dict(step)
            else:
                raise ValueError(f'{step_key}')

    def plot_step(self, key):
        step = self.get_step(key)

        symbols = step['symbols']
        labels = step['labels']
        metadata = step['metadata']
        # breakpoint()

        if self.n_components == 2:
            if '2d_proj' in step:
                projection = step['2d_proj']
            else:
                projection = self.reducer.fit_transform(step['embedding'])
                step['2d_proj'] = projection
            
            df = pd.DataFrame(dict(
                x=projection[:, 0],
                y=projection[:, 1],
                label=labels, 
                audio_path=metadata['audio_path']
            ))
            fig = px.scatter(df, x='x', y='y', color='label',
                             symbol=symbols,
                             custom_data=['audio_path', 'label'],
                             **step['plotly_kwargs'])

        elif self.n_components == 3:
            if '3d_proj' in step:
                projection = step['3d_proj']
            else:
                projection = self.reducer.fit_transform(step['embedding'])
                step['3d_proj'] = projection

            df = pd.DataFrame(dict(
                x=projection[:, 0],
                y=projection[:, 1],
                z=projection[:, 2],
                label=labels, 
                audio_path=metadata['audio_path']
            ))
            fig = px.scatter_3d(df, x='x', y='y', z='z',
                                color='label', symbol=symbols, 
                                custom_data=['audio_path', 'label'],
                                **step['plotly_kwargs'])

        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        # fig.update_traces(customdata=metadata['audio_path'])
        return fig
