import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        #raise NotImplementedError()

        self.d_latent = d_latent
        self.n_tokens = n_tokens

        # Token embedding
        self.token_emb = torch.nn.Embedding(n_tokens, d_latent)

        # Learned "start" embedding used for the shift-by-1 (so first prediction depends on nothing from x)
        self.start_emb = torch.nn.Parameter(torch.zeros(d_latent))

        # A small decoder-only transformer implemented via causal mask on an encoder stack
        n_heads = 4
        n_layers = 4
        ff_dim = 4 * d_latent
        dropout = 0.0

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Project back to vocab
        self.to_logits = torch.nn.Linear(d_latent, n_tokens)

        # Helps training stability a bit
        self.final_norm = torch.nn.LayerNorm(d_latent)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        # True values are masked (disallowed). Shape (L, L).
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)    

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #raise NotImplementedError()

        """
        x: (B, h, w) integer tokens
        returns logits: (B, h, w, n_tokens)
        """
        if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            x = x.long()

        B, h, w = x.shape
        L = h * w
        device = x.device

        # Flatten tokens -> (B, L)
        x_seq = x.view(B, L)

        # Embed -> (B, L, d)
        emb = self.token_emb(x_seq)

        # Shift-by-1 AFTER embedding:
        # inputs to position t use [START] for t=0, and token_{t-1} for t>0
        start = self.start_emb.view(1, 1, self.d_latent).expand(B, 1, self.d_latent)
        emb_shifted = torch.cat([start, emb[:, :-1, :]], dim=1)  # (B, L, d)

        # Causal mask so position t can't attend to >t
        mask = self._causal_mask(L, device=device)

        # Transformer -> (B, L, d)
        z = self.transformer(emb_shifted, mask)
        z = self.final_norm(z)

        # Logits -> (B, L, n_tokens) -> (B, h, w, n_tokens)
        logits = self.to_logits(z).view(B, h, w, self.n_tokens)

        return logits, {}

    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        #raise NotImplementedError()

        """
        Autoregressively sample tokens left-to-right, top-to-bottom.
        Returns: (B, h, w) long
        """
        if device is None:
            device = next(self.parameters()).device

        L = h * w
        x = torch.zeros((B, L), dtype=torch.long, device=device)

        for t in range(L):
            logits, _ = self.forward(x.view(B, h, w))
            logits_t = logits.view(B, L, self.n_tokens)[:, t, :]  # (B, n_tokens)

            probs = torch.softmax(logits_t, dim=-1)
            sample = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

            x[:, t] = sample

        return x.view(B, h, w)
