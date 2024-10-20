from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import nltk
import tqdm

from utils.attention import MultiHeadAttention
from utils.decoder import DecoderBlock
from utils.transformer_model import TransformerDecoder
from dataset.text_dataset import TextDataset

nltk.download('punkt_tab')
text = []
data = 'input.txt'
DEVICE = 'cpu'

with open(data, 'r', encoding='utf-8') as f:
    for l in f:
        if l.strip() == '':
            continue
        text.append(l.strip())

sentences = sent_tokenize(' '.join(text), language='english')
train_sentences = sentences[:10000]
test_sentences = sentences[10000:]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
D_MODEL = 512
N_HEADS = 8
VOCAB_SIZE = len(tokenizer)
N_BLOCKS = 3
mha = MultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
decoder_block = DecoderBlock(D_MODEL, N_HEADS).to(DEVICE)
model = TransformerDecoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_BLOCKS).to(DEVICE).to(DEVICE)

train_dataset = TextDataset(train_sentences, tokenizer)
test_dataset = TextDataset(test_sentences, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)

loss_fnc = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
loss_epochs = []
for epoch in tqdm.tqdm(range(20)):
    loss_epoch = []
    for input_ids, labels_ids, data_attention_mask in tqdm.tqdm(train_loader):
        input_ids = input_ids.to(DEVICE)
        labels_ids = labels_ids.to(DEVICE)
        # pad_mask = pad_mask.to(DEVICE)
        data_attention_mask = data_attention_mask.to(DEVICE)

        preds = model(input_ids)
        preds = preds[data_attention_mask]
        labels_ids = labels_ids[data_attention_mask]

        optimizer.zero_grad()
        loss = loss_fnc(preds, labels_ids)
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())

    print(f'Loss_epoch {epoch + 1}: {np.mean(loss_epoch)}')
    loss_epochs.append(np.mean(loss_epoch))

torch.save(model, 'transformer.pt')