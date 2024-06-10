import os
import torch
from transformers import (
    get_linear_schedule_with_warmup,
)
from src.uc_api.entity import SentimentModelTrainerConfig
from src.uc_api.model.sentiment.sentiment_classifier import SentimentClassifier
from src.uc_api.model.sentiment.helper import SentimentHelper
from collections import defaultdict

from accelerate import Accelerator
accelerator = Accelerator()



class SentimentModelTrainer:

    def __init__(self, config: SentimentModelTrainerConfig):
        self.config = config

    def train_and_evaluate_sentiment_model(self):
        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading Train and Validation dataloaders
        data_loader_train = torch.load(os.path.join(self.config.transformation_root_dir, "train_dataloader.pt"))
        data_loader_validation = torch.load(os.path.join(self.config.transformation_root_dir, "validation_dataloader.pt"))

        # Raw BERT model picked with preset configurations
        model = SentimentClassifier(
            len(self.config.sentiment_class_dict.values()),
            self.config.pre_trained_model_name,
            self.config.dropout_parameter,
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        # Accelerate optimization
        data_loader_train, data_loader_validation, model, optimizer = accelerator.prepare(
            data_loader_train, data_loader_validation, model, optimizer,
            )        

        # number of epochs
        num_epochs = self.config.num_train_epochs

        # Number of steps for processing  
        total_steps_train = len(data_loader_train) * num_epochs      
        
        # Scheduler
        scheduler_train = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps_train,
            )
        
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss().to(device)        

        # Calling model trainer on dataloader
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)       
            
            # <=============== Executing the training module ===============>
            train_accuracy, train_loss = SentimentHelper.train_epoch(
                    model,
                    data_loader_train,
                    loss_fn,
                    optimizer,
                    device,
                    scheduler_train,
                    len(data_loader_train),
                    accelerator,
                )
            print(f'Train loss {train_loss}, accuracy {train_accuracy}')
            history['train_accuracy'].append(train_accuracy)
            history['train_loss'].append(train_loss)     

            # <=============== Executing the validation module ===============>
            validation_accuracy, validation_loss = SentimentHelper.eval_model(
                    model,
                    data_loader_validation,
                    loss_fn,
                    device,
                    len(data_loader_validation),)
            print(f'Validation loss {validation_loss}, accuracy {validation_accuracy}')            
            history['validation_accuracy'].append(validation_accuracy)
            history['test_loss'].append(validation_loss)

            if validation_accuracy and (validation_accuracy > best_accuracy):
                torch.save(model.state_dict(), self.config.model_ckpt)
                best_accuracy = validation_accuracy

            

            

        
        

        
