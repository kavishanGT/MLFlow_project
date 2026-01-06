import os
from Mlflow_project.constants import *
#from Mlflow_project.utils.common import read_yaml, create_directories
from Mlflow_project.entity.config_entity import (BaseModelConfig, DataIngestionConfig, TrainingConfig,EvaluationConfig)

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

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> BaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.model_trainer
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Brain_MRI_scan_images")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="resources/model_trainer/trained_model.h5",
            training_data="resources/data_ingestion/Brain_MRI_scan_images",
            mlflow_uri="https://dagshub.com/kavishanGT/MLFlow_project.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config