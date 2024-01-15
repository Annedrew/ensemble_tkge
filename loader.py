import torch
import sys
from tkge_models.de_simple import de_transe, de_simple, de_distmult, dataset, params
from tkge_models.TERO import TERO_model, Dataset

class Loader:
    def __init__(self, model_path, embedding):
        self.model_path = model_path
        self.embedding = embedding

    def load(self):
        old_modules = sys.modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            sys.modules['de_transe'] = de_transe
            sys.modules['de_simple'] = de_simple
            sys.modules['de_distmult'] = de_distmult
            sys.modules['dataset'] = dataset
            sys.modules['params'] = params
        elif self.embedding in ["TERO", "ATISE"]:
            sys.modules['model'] = TERO_model
            sys.modules['Dataset'] = Dataset
        model = torch.load(self.model_path, map_location="cpu")
        sys.modules = old_modules
        return model
    