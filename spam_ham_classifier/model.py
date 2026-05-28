import torch
import torch.nn as nn
import math

class Multi_Head_Attention(nn.Module):
    def __init__(self, din_, dout, dropout_r, numh, context_length, bias=False):
        super().__init__()
        assert dout % numh == 0, "dout must be divisible by numh"
        self.dout = dout
        self.heads = numh
        self.head_dim = dout // numh
        self.w_query = nn.Linear(din_, dout, bias=bias)
        self.w_key = nn.Linear(din_, dout, bias=bias)
        self.w_value = nn.Linear(din_, dout, bias=bias)
        self.out_proj = nn.Linear(dout, dout, bias=bias)
        self.dropout = nn.Dropout(dropout_r)
        # mask ek baar banao
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        batch, num_tokens, din_emb = x.shape
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        query = query.view(batch, num_tokens, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, num_tokens, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, num_tokens, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = query @ key.transpose(2, 3)
        # sqrt fix
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ value).transpose(1, 2)
        context = context.contiguous().view(batch, num_tokens, self.dout)
        return self.out_proj(context)


class Layer_Normalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(variance + self.eps) + self.shift


class Gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.44715 * x.pow(3))
        ))


class Feed_Forward_block(nn.Module):
    def __init__(self, no_emb):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(no_emb, 4 * no_emb, bias=True),
            Gelu(),
            nn.Linear(4 * no_emb, no_emb, bias=True)
        )

    def forward(self, x):
        return self.model(x)


class Transformer_Block(nn.Module):
    def __init__(self, Gpt_arch):
        super().__init__()
        self.attention = Multi_Head_Attention(
            din_=Gpt_arch['emb_dim'],
            dout=Gpt_arch['emb_dim'],
            numh=Gpt_arch['n_heads'],
            dropout_r=Gpt_arch['dropout_layer'],
            context_length=Gpt_arch['context_length'],
            bias=Gpt_arch['qkv_bias']
        )
        self.ff = Feed_Forward_block(Gpt_arch['emb_dim'])
        self.norm1 = Layer_Normalization(Gpt_arch['emb_dim'])
        self.norm2 = Layer_Normalization(Gpt_arch['emb_dim'])
        self.dropout = nn.Dropout(Gpt_arch['dropout_layer'])

    def forward(self, x):
        # Attention block — shortcut fix
        x = x + self.dropout(self.attention(self.norm1(x)))
        # FFN block — shortcut fix
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Quasar(nn.Module):
    def __init__(self, Gpt_arch):
        super().__init__()
        self.tok_emb = nn.Embedding(Gpt_arch['vocab_size'], Gpt_arch['emb_dim'])
        self.pos_emb = nn.Embedding(Gpt_arch['context_length'], Gpt_arch['emb_dim'])
        self.drop_emb = nn.Dropout(Gpt_arch['dropout_layer'])
        self.transformer_blocks = nn.Sequential(
            *[Transformer_Block(Gpt_arch) for _ in range(Gpt_arch['n_layers'])]
        )
        self.norm = Layer_Normalization(Gpt_arch['emb_dim'])
        self.out_head = nn.Linear(Gpt_arch['emb_dim'], Gpt_arch['vocab_size'], bias=False)

    def forward(self, x):
        batch, seq_len = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.drop_emb(tok + pos)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return self.out_head(x)