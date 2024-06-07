import os
import torch
from transformers import (
    get_linear_schedule_with_warmup,
)
from src.uc_api.entity import SentimentModelTrainerConfig
from src.uc_api.model.sentiment.sentiment_classifier import SentimentClassifier
from src.uc_api.model.sentiment.helper import SentimentHelper
from collections import defaultdict


class SentimentModelTrainer:

    def __init__(self, config: SentimentModelTrainerConfig):
        self.config = config

    def train_and_evaluate_sentiment_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Raw BERT model picked with preset configurations
        model = SentimentClassifier(
            len(self.config.sentiment_class_dict.values()),
            self.config.pre_trained_model_name,
            self.config.dropout_parameter,
        ).to(device)

        # Populating dataloader based as data split type
        data_loader_dict = {}
        for dataset_class in self.config.data_split_type:
            data_loader = torch.load(os.path.join(self.config.transformation_root_dir, f"{dataset_class}_dataloader.pt"))
            data_loader_dict[dataset_class] = data_loader
            data_loader_dict[f"{dataset_class}_len"] = sum([len(each) for each in data_loader])
        print(data_loader_dict["train_len"])


        # number of epochs
        epochs = self.config.num_train_epochs
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        # Number of steps for processing  
        total_steps = len(data_loader_dict["train"]) * epochs
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps,
            )
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # Calling model trainer on dataloader
        history = defaultdict(list)
        best_accuracy = 0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            if data_loader_dict.keys().__contains__("train"):
                train_acc, train_loss = SentimentHelper.train_epoch(
                    model,
                    data_loader_dict["train"],
                    loss_fn,
                    optimizer,
                    device,
                    scheduler,
                    data_loader_dict["train_len"],
                )
                print(f'Train loss {train_loss}, accuracy {train_acc}')
                history['train_acc'].append(train_acc)
                history['train_loss'].append(train_loss)

            if data_loader_dict.keys().__contains__("test"):
                test_acc, test_loss = SentimentHelper.eval_model(
                    model,
                    data_loader_dict["test"],
                    loss_fn,
                    device,
                    data_loader_dict["test_len"],)
                print(f'Test loss {test_loss}, accuracy {test_acc}')            
                history['test_acc'].append(test_acc)
                history['test_loss'].append(test_loss)

            if test_acc and (test_acc > best_accuracy):
                torch.save(model.state_dict(), self.config.model_ckpt)
                best_accuracy = test_acc
            # else:
            #     torch.save(model.state_dict(), self.config.model_ckpt)
            #     best_accuracy = train_acc

           

        
        

        
