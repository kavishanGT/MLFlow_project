import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from Mlflow_project.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        # Load model without restored optimizer state to avoid optimizer/variable mismatches
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
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
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
        
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Rebuild the model (clone) to ensure optimizer/variable sets are fresh
        try:
            weights = self.model.get_weights()
            self.model = tf.keras.models.clone_model(self.model)
            self.model.set_weights(weights)
        except Exception:
            # If cloning fails, proceed and attempt to recompile
            pass

        # Recompile the model with a fresh optimizer to avoid optimizer-variable mismatch
        try:
            from Mlflow_project.utils.common import read_yaml
            from Mlflow_project.constants import PARAMS_FILE_PATH
            params = read_yaml(PARAMS_FILE_PATH)
            lr = float(params.LEARNING_RATE)
        except Exception:
            lr = 0.01

        # Create optimizer and initialize its internal weights for current model variables
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
        try:
            opt._create_all_weights(self.model.trainable_variables)
        except Exception:
            pass

        # Try standard Keras training; if optimizer/variable mismatch persists, fall back to a manual loop
        try:
            self.model.compile(
                optimizer=opt,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"],
                run_eagerly=True
            )

            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator
            )

        except ValueError as e:
            # Known issue: optimizer/variable mismatch when loading a compiled model. Fall back to manual training loop.
            print('Optimizer-variable mismatch detected. Falling back to manual training loop.')
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            train_acc = tf.keras.metrics.CategoricalAccuracy()
            val_acc = tf.keras.metrics.CategoricalAccuracy()

            for epoch in range(self.config.params_epochs):
                print(f'Epoch {epoch+1}/{self.config.params_epochs}')
                train_acc.reset_states()
                steps = 0
                for _ in range(self.steps_per_epoch):
                    x_batch, y_batch = next(self.train_generator)
                    x_batch = tf.convert_to_tensor(x_batch)
                    y_batch = tf.convert_to_tensor(y_batch)
                    with tf.GradientTape() as tape:
                        preds = self.model(x_batch, training=True)
                        loss = loss_fn(y_batch, preds)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    opt.apply_gradients(zip(grads, self.model.trainable_variables))
                    train_acc.update_state(y_batch, preds)
                    steps += 1
                    if steps % 50 == 0:
                        print(f'  step {steps}/{self.steps_per_epoch} - train_acc: {train_acc.result().numpy():.4f}')

                # Validation pass (simple accuracy)
                val_acc.reset_states()
                for _ in range(self.validation_steps):
                    x_val, y_val = next(self.valid_generator)
                    preds = self.model(x_val, training=False)
                    val_acc.update_state(y_val, preds)

                print(f'  epoch {epoch+1} - train_acc: {train_acc.result().numpy():.4f}, val_acc: {val_acc.result().numpy():.4f}')

        # Save final model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )