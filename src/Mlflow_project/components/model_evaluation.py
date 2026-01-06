import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from Mlflow_project.entity.config_entity import EvaluationConfig

from pathlib import Path
import sys, os

# Ensure we're using the project root and add the absolute `src` path to sys.path
project_root = Path.cwd()
src_path = str(project_root / "src")
print("CWD:", project_root)
print("Adding to sys.path:", src_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Quick verification
import pkgutil
print("Mlflow_project present in src?", any(p.name == 'Mlflow_project' for p in pkgutil.iter_modules([src_path])))

# Now import
from Mlflow_project.constants import *
from Mlflow_project.utils.common import read_yaml, create_directories, save_json
print("Imported Mlflow_project successfully.")


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            # if tracking_url_type_store != "file":
                
            #     mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            # else:
            #     mlflow.keras.log_model(self.model, "model")
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="VGG16Model",
                    # keras_model_kwargs={
                    #     "save_format": "keras"
                    # }
                )
            else:
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    # keras_model_kwargs={
                    #     "save_format": "keras"
                    # }
                )