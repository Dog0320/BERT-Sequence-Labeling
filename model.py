import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]

        output = self.dropout(sequence_output)
        output = self.classifier(output)

        return output
