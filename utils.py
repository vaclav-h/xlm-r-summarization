import torch
from sentence_splitter import SentenceSplitter
from torch.nn.functional import pad
from nltk import ngrams
from dataset import XLMRobertaExtDataset


class ExtEvaluator:
    def __init__(self, fpath, eval_size=None, pos_embed=512, eval_bs=32, model_type="xlm-roberta-large"):
        """
        fpath : path to the evaluation dataset built using build_ext_dataset.py
        eval_size : take only the first eval_size elements of the dataset
        """
        self.dataset = XLMRobertaExtDataset(fpath, pos_embed=pos_embed, model_type=model_type)
        if eval_size:
            self.dataset = torch.utils.data.Subset(self.dataset, range(eval_size))
        self.dloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                   batch_size=eval_bs,
                                                   shuffle=False)
        self.loss_fn = torch.nn.BCELoss(reduction='none')
        
    def __call__(self, model):
        model.eval()
        
        total_loss = 0.0
        for batch in self.dloader:
            batch = {k: v.squeeze(1).to('cuda') for k, v in batch.items()}
            with torch.no_grad():
                scores = model(batch['input_ids'],
                               batch['attention_mask'],
                               batch['cls_ids'],
                               batch['cls_mask'])
                # Compute loss
                loss = self.loss_fn(scores, batch['labels'])
                loss = (loss * batch['cls_mask']).sum() / loss.numel()
                total_loss += loss.item()
        total_loss /= len(self.dloader)
        
        model.train()
        return total_loss    


def extract(model, tokenizer, text, n_sent=3, max_src_sentences=32, max_length=512,
            splitter=None, trigram_block=True, join=True, lang='cs'):
    """
    Method to generate extractive summary using fine-tuned model
    
    Args:
        model: XLMRobertaExtModel
        tokenizer: XLMRobertaTokenizerFast
        text: str - input document
        n_sent: int - number of sentences in the summary
        max_src_sentences: int - number of sentences from the source to consider
        max_length: int - number of positional embeddings in model
        splitter: SentenceSplitter
        trigram_block: bool - whether or not to include sentences with common trigram
        join: bool - whether or not to join the summary into single string
        lang: str - language of text, eg. 'en', 'de', 'cs', ...
    Returns:
        str - if join is True
        list(str) if join is False
    """
    device = next(model.parameters()).device.type
    
    if not splitter:
        splitter = SentenceSplitter(language=lang)
    
    src_split = splitter.split(text)
    src_prep = '</s><s>'.join(src_split)
    src_tok = tokenizer(src_prep, max_length=max_length,
                        truncation=True, padding='max_length', return_tensors='pt')

    cls_ids = (src_tok.input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
    if len(cls_ids) >= max_src_sentences:
        cls_ids = cls_ids[:max_src_sentences]
    else:
        cls_ids = pad(cls_ids, (0, max_src_sentences-len(cls_ids)), value=-1)
    cls_mask = (cls_ids != -1).float()
    
    scores = model(src_tok['input_ids'].to(device),
                   src_tok['attention_mask'].to(device),
                   cls_ids[None, :].to(device),
                   cls_mask[None, :].to(device))
    
    if trigram_block:
        indices = torch.topk(scores, n_sent+2, dim=1).indices.squeeze()
        incl_trigrams = set()
        top_indices = []
        for i in indices:
            curr_trigrams = set(ngrams(src_split[i].split(), 3))
            if len(incl_trigrams.intersection(curr_trigrams)) == 0:
                top_indices.append(i)
                incl_trigrams = incl_trigrams.union(curr_trigrams)
            if len(top_indices) == n_sent:
                top_indices = sorted(top_indices)
                break
    else:
        top_indices = torch.topk(scores, n_sent, dim=1).indices
        top_indices = torch.sort(top_indices).values.squeeze().tolist()

    if join:
        return " ".join([src_split[i] for i in top_indices])
    else:
        return [src_split[i] for i in top_indices]    
