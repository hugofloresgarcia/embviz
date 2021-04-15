# embedding viz

My purpose is to serve interactive html plotly graphs

made using dash!

to visualize the embedding spaces for a particular run in music trees:

1. run the embedding server:

```bash
python -m embviz.viz --dir /<PATH_TO_RUNS>/<NAME>/<VERSION>
```

2. if you want to listen to the embeddings, make sure to serve the data directory as well

(make sure this serves at `http://0.0.0.0:8000`)
```bash
python -m http.server /<PATH_TO_DATA>/
```