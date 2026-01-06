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

from Mlflow_project.config.configuration import ConfigurationManager
from Mlflow_project.components.model_evaluation import Evaluation
from Mlflow_project import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            