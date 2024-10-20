from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, tok, max_length = 30):
        self.data = data
        self.tok = tok
        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        data_tokens = self.tok(data, return_tensors = 'pt', max_length = self.max_length, truncation=True, padding = 'max_length')

        data_input_ids = data_tokens['input_ids'][0]
        data_attention_mask = data_tokens['attention_mask'][0]

        input_ids = data_input_ids[:-1]
        labels_ids = data_input_ids[1:]

        return input_ids, labels_ids, data_attention_mask[1:].bool()