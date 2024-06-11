import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_from_disk 
from src.uc_api.logging import logger
from src.uc_api.entity import SentimentDataTransformationConfig


# BERT based encoding
class SentimentDataset(Dataset):
    def __init__(self, text, sentimentRating, tokenizer, max_length):
        self.text = text
        self.sentimentRating = sentimentRating
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sentimentRating": torch.tensor(
                self.sentimentRating[item], dtype=torch.long
            ),
        }


class DataTransformation:
    def __init__(self, config: SentimentDataTransformationConfig):
        self.config = config

    def transform_and_save_to_file(self):
        # loading the dataloder for transformation
        dataset = load_from_disk(self.config.dataset_path)
        # print(data_set)
        
        for dataset_class in self.config.data_split_type:
            ds = SentimentDataset(

                # feature
                text=dataset[dataset_class][self.config.feature_column_name],

                # Label in negative: 0, neutral: 1, positive: 2
                sentimentRating=[self.config.sentiment_class_dict[element] for element in dataset[dataset_class][self.config.label_column_name]],
                tokenizer=BertTokenizer.from_pretrained(
                    self.config.tokenizer_name,
                ),
                # 
                max_length=self.config.max_length,
            )
            # Saving the dataloader based on the data split type
            torch.save( 
                DataLoader(
                    ds,
                    batch_size=self.config.batch_size,
                ),
                os.path.join(self.config.root_dir, f"{dataset_class}_dataloader.pt"),
            )
            logger.info(
                f"""Successfully saved {dataset_class} dataloader to {self.config.root_dir}"""
            )