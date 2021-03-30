import os
import glob
from pathlib import Path
import json

def safe_string(string: str):
    """https: / stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string"""
    keepcharacters = (' ', '.', '_')
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip()

"""
glob
"""
def glob_all_entries(root_dir, pattern='**/*.json'):
    """ reads all metadata files recursively and loads them into
    a list of dicts
    """
    pattern = os.path.join(root_dir, pattern)
    filepaths = glob.glob(pattern, recursive=True)
    records = [load_entry(path) for path in filepaths]
    return records

"""
json and yaml 
"""
def _add_file_format_to_filename(path: str, file_format: str):
    if '.' not in file_format:
        file_format = f'.{file_format}'

    if Path(path).suffix != file_format:
        path = Path(path).with_suffix(file_format)
    return str(path)

def save_entry(entry, path, format='json'):
    os.makedirs(Path(path).parent, exist_ok=True)
    path = _add_file_format_to_filename(path, format)
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(entry, f)

def load_entry(path, format='json'):
    entry = None
    if format == 'json':
        with open(path, 'r') as f:
            entry = json.load(f)
    else:
        raise ValueError(f'unsupported format: {format}')

    return entry
