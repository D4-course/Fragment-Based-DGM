"""Filesystem"""
import json
import pickle as pkl
import sh
import pandas as pd

def load_pickle(path):
    """load_pickle"""
    return pkl.load(open(path, "rb"))


def save_pickle(obj, path):
    """save_pickle"""
    pkl.dump(obj, open(path, "wb"))


def load_json(path):
    """load_json"""
    return json.load(open(path, "r"))


def save_json(obj, path):
    """save_json"""
    json.dump(obj, open(path, "w"), indent=2)


def commit(experiment_name, time):
    """
    Try to commit repo exactly as it is when starting
    the experiment for reproducibility.
    """
    try:
        sh.git.commit('-a',
                      m=f'"auto commit tracked '
                        f'files for new experiment: '
                        f'{experiment_name} on {time}"',
                      allow_empty=True
                      )
        commit_hash = sh.git('rev-parse', 'HEAD').strip()
        return commit_hash
    except Exception:
        return '<Unable to commit>'


def load_dataset(config, kind):
    """load_dataset"""
    assert kind in ['train', 'test']
    path = config.path('data')
    print(path)
    filename = path / f'{kind}.smi'
    return pd.read_csv(filename, index_col=0)
