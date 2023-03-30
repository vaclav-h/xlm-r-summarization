import torch
from transformers import XLMRobertaModel
from torch import nn


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        h = self.linear(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask
        return sent_scores


class XLMRobertaExtModel(nn.Module):
    def __init__(self, checkpoint=None, pos_embed=512, model_type="xlm-roberta-large"):
        super(XLMRobertaExtModel, self).__init__()
        
        self.roberta = XLMRobertaModel.from_pretrained(model_type)
        
        # Copy pretrained part of embeddings and randomly initialize the rest
        if pos_embed > 512:
            my_pos_embeddings = nn.Embedding(pos_embed+2, self.roberta.config.hidden_size, padding_idx=1)
            my_pos_embeddings.weight.data[:512+2] = self.roberta.embeddings.position_embeddings.weight.data
            self.roberta.embeddings.position_embeddings = my_pos_embeddings
            self.roberta.embeddings.token_type_ids = torch.zeros((1, pos_embed+2), dtype=torch.long)
            self.roberta.config.max_position_embeddings = pos_embed+2

        self.clasifier = Classifier(self.roberta.config.hidden_size)

        if checkpoint is not None:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(checkpoint))
            else:
                self.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def forward(self, src_ids, src_mask, cls_ids, cls_mask):
        """
        src_ids  : [batch_size, model_max_length]
        src_mask : [batch_size, model_max_length]
        cls_ids  : [batch_size, max_src_sentences]
        cls_mask : [batch_size, max_src_sentences]
        
        returns scores : [batch_size, max_src_sentences]
        """
        
        roberta_vec = self.roberta(src_ids, src_mask).last_hidden_state
        sents_vec = roberta_vec[torch.arange(roberta_vec.size(0)).unsqueeze(1), cls_ids]
        sents_vec = sents_vec * cls_mask[:, :, None] # [batch_size, max_src_sentences, model_hidden_size]
        sent_scores = self.clasifier(sents_vec, cls_mask)
        return sent_scores
