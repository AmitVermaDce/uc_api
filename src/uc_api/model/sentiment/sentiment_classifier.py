import torch
from transformers import BertModel


class SentimentClassifier(torch.nn.Module):

    def __init__(
        self,
        number_of_classes,
        pre_trained_model_name,
        dropout_parameter,
    ):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            pre_trained_model_name,
            return_dict=False,
        )
        self.drop = torch.nn.Dropout(p=dropout_parameter)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, number_of_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = self.drop(pooled_output)
        return self.out(output)
