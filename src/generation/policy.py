"""Complete-sequence causal policies with identical legal action support."""
import math
import torch
from torch import nn
AA = 'ACDEFGHIKLMNPQRSTVWY'
PAD, BOS, EOS, SEP = 20, 21, 22, 23

def encode(sequence):
    if not 10 <= len(sequence) <= 50 or any(a not in AA for a in sequence):
        raise ValueError('Sequences must have 10–50 standard residues')
    return [AA.index(a) for a in sequence]

class Policy(nn.Module):
    def __init__(self, hidden=256, heads=8, layers=4, feedforward=1024, dropout=0.1):
        super().__init__()
        self.config = dict(hidden=hidden, heads=heads, layers=layers, feedforward=feedforward, dropout=dropout)
        self.embedding = nn.Embedding(24, hidden)
        self.position = nn.Embedding(104, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 24)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        h = self.embedding(tokens) + self.position(positions)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        return self.output(self.decoder(h, mask=causal, src_key_padding_mask=tokens.eq(PAD)))

    def prefix(self, parent=None):
        return [BOS] if parent is None else [BOS] + encode(parent) + [SEP]

    def legal_logits(self, logits, length):
        allowed = torch.zeros(24, dtype=torch.bool, device=logits.device)
        allowed[:20] = length < 50
        allowed[EOS] = length >= 10
        return logits.masked_fill(~allowed, -torch.inf)

    def sample(self, parent=None, top_p=1.0, temperature=1.0):
        if not 0 < top_p <= 1 or not math.isfinite(temperature) or temperature <= 0: raise ValueError('Invalid sampling settings')
        tokens = self.prefix(parent)
        residues = []
        device = next(self.parameters()).device
        for length in range(51):
            logits = self.legal_logits(self(torch.tensor([tokens], device=device))[0, -1], length) / temperature
            if top_p < 1:
                sorted_logits, indices = logits.sort(descending=True)
                remove = sorted_logits.softmax(-1).cumsum(-1) - sorted_logits.softmax(-1) >= top_p
                logits = logits.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
            token = torch.distributions.Categorical(logits=logits).sample().item()
            tokens.append(token)
            if token == EOS: break
            residues.append(AA[token])
        return ''.join(residues)

    def log_probability(self, sequence, parent=None):
        prefix = self.prefix(parent)
        targets = encode(sequence) + [EOS]
        device = next(self.parameters()).device
        logits = self(torch.tensor([prefix + targets[:-1]], device=device))[0]
        return torch.stack([self.legal_logits(logits[len(prefix)-1+i], i).log_softmax(-1)[token] for i, token in enumerate(targets)]).sum()

    def supervised_loss(self, sequences, parents=None):
        parents = parents if parents is not None else [None] * len(sequences)
        if not sequences or len(parents)!=len(sequences):raise ValueError('Nonempty aligned sequences and parents required')
        inputs, labels = [], []
        for sequence, parent in zip(sequences, parents):
            prefix, target = self.prefix(parent), encode(sequence) + [EOS]
            inputs.append(prefix + target[:-1])
            labels.append([-100] * (len(prefix)-1) + target)
        width = max(map(len, inputs)); device = next(self.parameters()).device
        x = torch.tensor([v + [PAD]*(width-len(v)) for v in inputs], device=device)
        y = torch.tensor([v + [-100]*(width-len(v)) for v in labels], device=device)
        return nn.functional.cross_entropy(self(x).flatten(0,1), y.flatten(), ignore_index=-100)

def reinforce_loss(policy, reference, sequences, rewards, beta, parent=None):
    logp = torch.stack([policy.log_probability(s, parent) for s in sequences])
    with torch.no_grad():
        logref = torch.stack([reference.log_probability(s, parent) for s in sequences])
        returns = logp.detach().new_tensor(rewards) - beta * (logp.detach() - logref)
        baseline = (returns.sum()-returns)/(len(returns)-1) if len(returns)>1 else torch.zeros_like(returns)
    return -((returns-baseline).detach()*logp).mean()
