# models.py
"""
Neural network architecture and loss function for the FIFA World Cup
xG prediction model.
"""

import torch
import torch.nn as nn


class PoissonLoss(nn.Module):
    """Poisson negative log‑likelihood loss with optional sample weights."""

    def forward(self, pred, target, weights=None):
        # Clamp to avoid log(0) and stabilise gradients
        pred = torch.clamp(pred, min=0.1, max=6.0)
        loss = pred - target * torch.log(pred + 1e-8)
        loss = loss.sum(dim=1)               # sum over home/away goals
        if weights is not None:
            loss = loss * weights
        return loss.mean()


class OrderInvariantPredictor(nn.Module):
    """
    Predicts home/away expected goals (xG) from:
      - team embeddings
      - sequence of last HISTORY_LEN matches (transformer encoder)
      - 7 hand‑crafted static features (Elo diff, squad value diffs, etc.)

    The architecture is symmetric with respect to swapping teamA/teamB
    thanks to two‑pass averaging during training (implemented in main).
    """

    def __init__(self, num_teams, embed_dim=16, hist_len=10,
                 hist_input_dim=4, static_dim=7):
        super().__init__()

        # Team identity embedding
        self.team_embedding = nn.Embedding(num_teams, embed_dim)
        self.emb_dropout = nn.Dropout(0.1)

        # Sequence encoder for recent match history
        self.hist_proj = nn.Linear(hist_input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, batch_first=True, dropout=0.2
        )
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Static feature branch
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Fusion of all representations
        fusion_dim = embed_dim * 2 + 32 * 2 + 16
        self.final_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 2),          # two outputs: xG home, xG away
            nn.Softplus(),             # ensures positive outputs
        )

    def forward(self, teamA_id, teamB_id, teamA_seq, teamB_seq, static):
        # Embed team identities
        emb_A = self.emb_dropout(self.team_embedding(teamA_id))
        emb_B = self.emb_dropout(self.team_embedding(teamB_id))

        # Encode recent form (take the last hidden state)
        state_A = self.hist_encoder(self.hist_proj(teamA_seq))[:, -1, :]
        state_B = self.hist_encoder(self.hist_proj(teamB_seq))[:, -1, :]

        # Static feature embedding
        static_out = self.static_branch(static)

        # Concatenate everything and feed to final MLP
        combined = torch.cat([emb_A, emb_B, state_A, state_B, static_out], dim=1)
        return self.final_mlp(combined)