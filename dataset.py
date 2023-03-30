import json
import torch
import bisect
from datasets import Dataset
from transformers import XLMRobertaTokenizerFast
from torch.nn.functional import pad


class XLMRobertaExtDataset(Dataset):
    def __init__(self, fpath, n_samples=None, max_src_sentences=32,
                 pos_embed=512, model_type="xlm-roberta-large"):
        """
        fpath : path to jsonl file with dataset built using build_ext_dataset.py 
        """

        self.samples = []
        with open(fpath) as f:
            for line in f:
                self.samples.append(json.loads(line))
                
        if n_samples is not None:
            self.samples = self.samples[:n_samples]
        
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_type)
        self.enc_len = pos_embed
        self.max_src_sentences = max_src_sentences
    
    def get_texts(self):
        texts = [self.samples[i]['text'] for i in range(len(self))]
        return texts
   
    def get_abstracts(self):
        abstracts = [self.samples[i]['abstract'] for i in range(len(self))]
        return abstracts
    
    def get_targets(self):
        targets = [self.samples[i]['tgt_sentences'] for i in range(len(self))]
        return targets
    
    def __len__(self):
        return len(self.samples)
    
    def _pad2max_sent(self, x):
        if len(x) >= self.max_src_sentences:
            return x[:self.max_src_sentences]
        return pad(x, (0,self.max_src_sentences-len(x)), value=-1)
    
    def __getitem__(self, idx):
        # Preprocess input text
        src_prep = '</s><s>'.join(self.samples[idx]['text'])
        
        # Tokenize text
        src_tok = self.tokenizer(src_prep, max_length=self.enc_len, truncation=True, padding='max_length', return_tensors='pt')
        
        # Get ids of classification (<s>) tokens and pad to max_src_sentences
        cls_ids = (src_tok.input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
        cls_ids = self._pad2max_sent(cls_ids)
        cls_mask = (cls_ids != -1).float()
        
        # Encode targets
        tgts = self.samples[idx]['tgt_sentences']
        max_sent_label_id = bisect.bisect_left(tgts, self.max_src_sentences)
        labels = torch.zeros(self.max_src_sentences)
        labels[tgts[:max_sent_label_id]] = 1.
             
        # Prepare sample
        sample = {'input_ids' : src_tok['input_ids'],
                  'attention_mask' : src_tok['attention_mask'],
                  'cls_ids' : cls_ids.unsqueeze(0),
                  'cls_mask' : cls_mask.unsqueeze(0),
                  'labels' : labels.unsqueeze(0)}
      
        return sample
