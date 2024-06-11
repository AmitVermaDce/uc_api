import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.model.sentiment.sentiment_classifier import SentimentClassifier


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_sentiment_prediction_config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.tokenizer_name,
        )

        classifier = SentimentClassifier(
            len(self.config.sentiment_class_dict.values()),
            self.config.pre_trained_model_name,
            self.config.dropout_parameter,
        )
        classifier.load_state_dict(
            torch.load(
                self.config.model_ckpt,
                map_location=self.device,
            )
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict_sentiment(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.config.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(
                self.classifier(
                    input_ids,
                    attention_mask,
                ),
                dim=1,
            )
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_sentiment_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            [k for k, v in self.config.sentiment_class_dict.items() if v==predicted_sentiment_class][0],
            confidence.item(),
            dict(zip(list(self.config.sentiment_class_dict.keys()) ,probabilities)),           
        )


sentiment_prediction_pipeline = PredictionPipeline()


def get_sentiment_model():
    return sentiment_prediction_pipeline
