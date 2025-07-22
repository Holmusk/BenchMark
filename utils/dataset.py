from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, batch_size=1000):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.encodings = None
        self._tokenize_all()
    
    def _tokenize_all(self):
        # Tokenize in batches to avoid memory issues
        all_input_ids = []
        all_attention_mask = []
        
        for i in tqdm(range(0, len(self.texts), self.batch_size), desc="Tokenizing batches"):
            batch_texts = self.texts[i:i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            all_input_ids.append(encodings['input_ids'])
            all_attention_mask.append(encodings['attention_mask'])
        
        # Concatenate all batches
        self.encodings = {
            'input_ids': torch.cat(all_input_ids),
            'attention_mask': torch.cat(all_attention_mask)
        }
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids']) 