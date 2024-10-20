import torch
from transformers import GPT2Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('transformer.pt')
model_llava = torch.load('transformer_RoPE.pt')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

@torch.no_grad()
def generate_batch(model, ids, is_decode = False, max_length = 20):

    model.eval()
    if type(ids) == list:
        symbols = torch.tensor(ids, device = DEVICE).view(len(ids),1)

    else:
        symbols = torch.tensor(ids, device = DEVICE).view(1,len(ids))

    for _ in range(max_length - 1):

        cur_symbols = model(symbols).argmax(-1)[:, -1][:, None]
        symbols = torch.cat([symbols, cur_symbols], dim = 1)

    if is_decode:
        symbols = tokenizer.batch_decode(symbols, skip_special_tokens=True)
    return symbols

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
D_MODEL = 256
N_HEADS = 8
max_length = 30
VOCAB_SIZE = len(tokenizer)

import tkinter as tk

def update_suggestions(event):
    typed_text = search_entry.get()
    matching_suggestions = [[], []]

    data_tokens = tokenizer(typed_text, return_tensors = 'pt', max_length = max_length, truncation=True, padding = 'max_length')

    data_input_ids = torch.tensor(data_tokens['input_ids'][0][data_tokens['input_ids'][0] != tokenizer(tokenizer.eos_token)['input_ids'][0]])

    print(typed_text)
    if typed_text != '':
        text = generate_batch(model, ids = data_input_ids, is_decode=True, max_length=2)
        text_RoPE = generate_batch(model_llava, ids = data_input_ids, is_decode=True, max_length=2)
        print(text)
        if text not in matching_suggestions:
            matching_suggestions[0].append(text[0])
            matching_suggestions[1].append(text_RoPE[0])

    suggestion_box.delete(0, tk.END)
    suggestion_box_.delete(0, tk.END)

    for suggestion in matching_suggestions[0]:
        suggestion_box.insert(tk.END, suggestion)
    for suggestion in matching_suggestions[1]:
        suggestion_box_.insert(tk.END, suggestion)

    if matching_suggestions[0]:
        suggestion_box.pack(pady=5)
    else:
        suggestion_box.pack_forget()

    if matching_suggestions[1]:
        suggestion_box_.pack(pady=5)
    else:
        suggestion_box_.pack_forget()

def select_suggestion(event):
    selected = suggestion_box.get(tk.ACTIVE)
    search_entry.delete(0, tk.END)
    search_entry.insert(0, selected)
    suggestion_box.pack_forget()
    update_suggestions(event)

def select_suggestion_(event):
    selected = suggestion_box_.get(tk.ACTIVE)
    search_entry.delete(0, tk.END)
    search_entry.insert(0, selected)
    suggestion_box_.pack_forget()
    update_suggestions(event)

root = tk.Tk()
root.title("Поисковая строка с предложениями")
root.geometry("600x700")

search_entry = tk.Entry(root, width=60)
search_entry.pack(pady=10)
search_entry.bind("<KeyRelease>", update_suggestions)

label_box = tk.Label(root, width=60, text='Calssic PE')
label_box.pack(pady=10)

suggestion_box = tk.Listbox(root, width=60)
suggestion_box.bind("<<ListboxSelect>>", select_suggestion)
suggestion_box.pack(pady=10)

label_box_ = tk.Label(root, width=60, text='RoPE')
label_box_.pack(pady=10)

suggestion_box_ = tk.Listbox(root, width=60)
suggestion_box_.bind("<<ListboxSelect>>", select_suggestion_)
suggestion_box_.pack(pady=10)

root.mainloop()
