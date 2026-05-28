import torch
import torch.nn as nn
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
from .model import Quasar

Gpt_arch_cpu = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "dropout_layer": 0.1,   # ← 0.1 rakho — model.eval() handle kar lega
    "n_layers": 12,
    "qkv_bias": True
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    # Load checkpoint robustly (works if file contains a state_dict or a full checkpoint)
    checkpoint = torch.load(path, map_location='cpu')

    model = Quasar(Gpt_arch_cpu)
    # Replace language-model head with a binary classifier head for spam/ham
    model.out_head = nn.Linear(Gpt_arch_cpu['emb_dim'], 2, bias=True)

    # Determine where the state_dict is inside the checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load weights; use strict=False so head size mismatches won't error out
    load_result = model.load_state_dict(state_dict, strict=False)
    print("load_state_dict result:", load_result)

    model = model.to(device)
    model.eval()

    # Quick sanity prints (non-fatal)
    try:
        print("out_head weights:", model.out_head.weight[0][:3])
        print("First transformer weight:", 
              model.transformer_blocks[0].attention.w_query.weight[0][:3])
    except Exception as e:
        print("Warning printing weights failed:", e)

    return model
Model=load_model(r'spam_ham_classifier\rex.pth')

def classify_review(text, model=Model, tokenizer=tokenizer, device=device, max_length=257, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    predicted_label = torch.argmax(logits, dim=-1).item()
    label = "spam" if predicted_label == 1 else "not spam"

    # Return label and a plain Python list of probabilities for UI use
    try:
        probs_list = probs.squeeze(0).cpu().tolist()
    except Exception:
        probs_list = [float(p) for p in probs.cpu().numpy().ravel().tolist()]

    return label, probs_list

        # Clean outpu