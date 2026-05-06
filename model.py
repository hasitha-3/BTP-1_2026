import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset

MAX_OBSTACLES = 10

class QuotientSpaceDataset(Dataset):
    """
    Loads dataset.json and returns model-ready tensors.
    State vector  : [start(4), goal(4), dof/4] = 9-D
    Obstacle token: [x, y, z, radius]          = 4-D per obstacle
    Padding mask  : True = padded token (ignore in attention)
    Label         : infeasibility_link — 0=Feasible, 1-4=first failing link
    """

    def __init__(self, json_file, max_obstacles=MAX_OBSTACLES):
        print(f"Loading dataset from {json_file}...")
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")
        self.max_obstacles = max_obstacles

    def __len__(self):
        return len(self.data)

    def _pad_to_4(self, arr):
        out = np.zeros(4, dtype=np.float32)
        out[:len(arr)] = arr
        return out

    def __getitem__(self, idx):
        item = self.data[idx]
        start    = self._pad_to_4(item['start_config'])
        goal     = self._pad_to_4(item['goal_config'])
        dof_norm = np.array([item['dof'] / 4.0], dtype=np.float32)
        state    = np.concatenate([start, goal, dof_norm])
        obs_t = torch.zeros((self.max_obstacles, 4), dtype=torch.float32)
        obs   = item['obstacles']
        n     = min(len(obs), self.max_obstacles)
        for i in range(n):
            pos    = obs[i]['position']
            obs_t[i] = torch.tensor(
                [pos[0], pos[1], pos[2], obs[i]['radius']],
                dtype=torch.float32
            )

        mask = torch.ones(self.max_obstacles, dtype=torch.bool)
        if n > 0:
            mask[:n] = False
        else:
            mask[0] = False  

        target = torch.tensor(int(item['infeasibility_link']), dtype=torch.long)

        return {
            'state':        torch.tensor(state, dtype=torch.float32),
            'obstacles':    obs_t,
            'padding_mask': mask,
            'target':       target
        }


class QuotientTransformer(nn.Module):
    """
    Quotient-Space Transformer for robot planning feasibility prediction.
    Inputs:
      state        (B, 9)              : [start(4), goal(4), dof_norm(1)]
      obstacles    (B, max_obs, 4)     : [x, y, z, radius] per obstacle
      padding_mask (B, max_obs)        : True = padded / ignore
    Output:
      logits       (B, 5)              : class scores 0-4
    Architecture:
      A. Obstacle encoder : Linear(4→d) + TransformerEncoder + mean-pool
      B. State encoder    : MLP(9→32→d)
      C. Classifier       : MLP(2d→64→5)
    """

    def __init__(self, d_model=64, n_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.obs_embedding = nn.Linear(4, d_model)
        self.transformer   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)
        )

    def forward(self, state, obstacles, padding_mask):
        obs_emb = self.obs_embedding(obstacles)
        tf_out  = self.transformer(obs_emb, src_key_padding_mask=padding_mask)
        valid        = (~padding_mask).unsqueeze(-1).float()
        env_context  = (tf_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        state_feat = self.state_encoder(state)
        logits     = self.classifier(torch.cat([env_context, state_feat], dim=1))
        return logits