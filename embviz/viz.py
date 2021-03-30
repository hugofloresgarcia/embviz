from collections import OrderedDict
from pathlib import Path
import argparse
import glob
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.io as pio

from embviz.logger import EmbeddingSpaceLogger

def read_figures_from_path(directory: str):
    paths = glob.glob(str(Path(args.path) / '*.json'))
    figures = {}
    for fig_path in paths:
        try:
            idx = int(str(Path(fig_path).stem))
        except:
            raise ValueError(
                'the names of the .json files must be valid integers')
        with open(fig_path, 'r') as f:
            fig = pio.from_json(f.read())
        figures[idx] = fig

    # sort by integer key
    figures = OrderedDict(sorted(figures.items(), key=lambda x: x[0]))
    if len(figures) == 0:
        raise ValueError(f'path is empty: {args.path}')

    return figures

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

N_COMPONENTS = 2
METHOD = 'tsne'

logger = EmbeddingSpaceLogger(args.path, N_COMPONENTS, METHOD)

if len(logger.keys()) == 0:
    raise ValueError(f'no embeddings found in {args.path}')

app.layout = html.Div([

    dcc.Markdown(f"""
        # embedding space visualizer

        showing embeddings for {str(args.path)}
    """),
        
    html.Div([
        html.Div([
            dcc.Graph(figure=logger.plot_step(list(logger.keys())[0]),
                      id='graph-with-slider',
                      style={'width': '90vh', 'height': '90vh'}),
            html.Div([
                dcc.Slider(
                    id='step-slider',
                    min=0,
                    max=len(logger.keys()),
                    value=0,
                    marks={i: str(k) for i, k in enumerate(logger.keys())},
                    step=None,
                ),
            ]),
        ], className='six columns'),
        
        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
                Listen to each point's audio below.
            """),
            html.Pre(id='click-data', style=styles['pre']),
            # html.Div(id="placeholder", style={"display": "none"}),
            html.Audio(id="player", src=None, controls=True,),
            #    style={"width": "100%"},),
        ], className='six columns'),
    ], className='row'),
])


@app.callback(
    Output('click-data', 'children'),
    Input('graph-with-slider', 'clickData'))
def display_click_data(metadata):
    """ display the point's metadata on click"""
    try:
        return json.dumps(metadata['points'], indent=2)
    except:
        return None

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('step-slider', 'value'))
def update_figure(key):
    fig = logger.plot_step(key)
    return fig

@app.callback(
    Output('player', 'src'),
    Input('graph-with-slider', 'clickData'))
def attempt_load_audio(metadata: dict):
    try:
        print(metadata['points'])
        src = metadata['points'][0]['customdata'][0]
        src = Path(src).relative_to('/home/hugo/lab/music-trees/data/')
        src = 'http://0.0.0.0:8000/'+ str(src)
        return src
    except:
        return None

app.run_server(debug=True)
