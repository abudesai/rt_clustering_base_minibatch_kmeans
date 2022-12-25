import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")

model_fname = "model.save"

MODEL_NAME = "clustering_base_mini_batch_kmeans"


class ClusteringModel:
    def __init__(self, K, init="k-means++", verbose=False, **kwargs) -> None:
        self.K = K
        self.init = init
        self.verbose = verbose
        self.cluster_centers = None
        self.feature_names_in_ = None

        self.model = self.build_model()

    def build_model(self):
        model = MiniBatchKMeans(
            n_clusters=self.K,
            init=self.init,
            max_no_improvement=10,
            verbose=self.verbose,
            random_state=0,
        )
        return model

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.model.predict()
        # same for fit(), transform(), fit_transform(), etc.
        attr = getattr(self.model, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            return getattr(self.model, attr.__name__)(*args, **kwargs)

        return wrapper

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self.model, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = ClusteringModel.load(model_path)
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?"""
        )
    return model
