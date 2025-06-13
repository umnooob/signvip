import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoLengthDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(file_path, sep="\t")
        self.texts = self.data["SENTENCE"].values
        self.targets = (
            self.data["END_REALIGNED"].values - self.data["START_REALIGNED"].values
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.float),
        }
