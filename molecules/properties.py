"""Properties"""
import pandas as pd
from joblib import Parallel, delayed

from rdkit.Chem import Crippen, QED

from .conversion import mols_from_smiles
from .sascorer.sascorer import calculateScore

# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=unused-variable

def logp(mol):
    """logp"""
    return Crippen.MolLogP(mol) if mol else None


def mr(mol):
    """mr"""
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    """qed"""
    return QED.qed(mol) if mol else None


def sas(mol):
    """sas"""
    return calculateScore(mol) if mol else None


def add_property(dataset, name, n_jobs):
    """add_property"""
    fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr}[name]
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    prop = pjob(delayed(fn)(mol) for mol in mols)
    new_data = pd.DataFrame(prop, columns=[name])
    return pd.concat([dataset, new_data], axis=1, sort=False)
