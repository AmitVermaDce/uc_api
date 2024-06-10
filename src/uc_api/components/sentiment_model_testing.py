import os
import torch
from src.uc_api.entity import SentimentModelTestingConfig
from src.uc_api.model.sentiment.sentiment_classifier import SentimentClassifier
from src.uc_api.model.sentiment.helper import SentimentHelper


class SentimentModelTesting:

    def __init__(self, config: SentimentModelTestingConfig):
        self.config = config

    def test_sentiment_model(self):
        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading Train and Validation dataloaders
        data_loader_test = torch.load(os.path.join(self.config.test_dataset_path, "test_dataloader.pt"))

        # Raw BERT model picked with preset configurations
        model = SentimentClassifier(
            len(self.config.sentiment_class_dict.values()),
            self.config.pre_trained_model_name,
            self.config.dropout_parameter,
        ).to(device)         
        
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # <=============== Executing the testing module ===============>
        test_accuracy, _ = SentimentHelper.eval_model(
                model,
                data_loader_test,
                loss_fn,
                device,
                len(data_loader_test),
            )
        print(f'Testing Accuracy: {test_accuracy.item()}')
            
            
               

            

            

            

        
        

        
