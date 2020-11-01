from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler, BertModel
from torch.nn import CrossEntropyLoss
import torch


class Model(BertPreTrainedModel):
    def __init__(self, config):
        self.config = config
        super(Model, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = 2
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None):

        sequence_output, cls_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        logits = self.output(cls_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        logits = logits.softmax(dim=1)
        return logits