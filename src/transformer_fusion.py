from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _safe_parse_evidence(ev: Any) -> Dict[str, Any]:
    """Parse evidence that may be dict already or a string like "{'A': 'B'}"."""
    if ev is None or (isinstance(ev, float) and np.isnan(ev)):
        return {}
    if isinstance(ev, dict):
        return ev
    s = str(ev).strip()
    if not s:
        return {}
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        # last resort (unsafe in general, but we keep it for compatibility with existing repo)
        try:
            parsed = eval(s)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def _normalize_feature_key(col_name: str) -> str:
    """Normalize raw column names for matching to S2.2 feature importance names."""
    return str(col_name).strip()


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (np.floating, float, np.integer, int)):
            return float(x)
        s = str(x).strip()
        if s.lower() in {"nan", "none", ""}:
            return default
        return float(s)
    except Exception:
        return default


def _infer_bn_columns(results_df_bn: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    """
    Return (accident_type_col, evidence_col, prob_col, lift_col, support_prob_col).

    Supports variants:
      - evidence dict column may be 'evidence' or 'evidence_dict'
      - support_prob may be 'support_prob' or 'support_rate'
    """
    cols = set(results_df_bn.columns)
    accident_type_col = "accident_type" if "accident_type" in cols else "Accident Type"
    evidence_col = "evidence"
    if "evidence_dict" in cols and "evidence" not in cols:
        evidence_col = "evidence_dict"
    prob_col = "prob" if "prob" in cols else "P"
    lift_col = "lift" if "lift" in cols else "Lift"
    support_prob_col = "support_prob" if "support_prob" in cols else ("support_rate" if "support_rate" in cols else "support")
    return accident_type_col, evidence_col, prob_col, lift_col, support_prob_col


def _build_importance_lookup(feature_importances_df: pd.DataFrame) -> Dict[str, float]:
    """
    Build mapping: raw_feature_name -> importance score.

    Supports column variants:
      - ('feature', 'importance')
      - ('feature_name', 'importance_score')
      - ('Unnamed: 0', 'importance_score')  [common in your new files]
    """
    df = feature_importances_df.copy()
    cols = list(df.columns)
    if "feature" in cols and "importance" in cols:
        feat_col, imp_col = "feature", "importance"
    elif "feature_name" in cols and "importance_score" in cols:
        feat_col, imp_col = "feature_name", "importance_score"
    elif "Unnamed: 0" in cols and "importance_score" in cols:
        feat_col, imp_col = "Unnamed: 0", "importance_score"
    else:
        # fallback: first col is feature, second is importance
        if len(cols) < 2:
            raise ValueError("feature_importances_df needs at least 2 columns (feature, importance).")
        feat_col, imp_col = cols[0], cols[1]

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        k = str(r[feat_col]).strip()
        out[k] = _coerce_float(r[imp_col], default=0.0)
    return out


def _importance_for_col(col_name: str, importance_lookup: Dict[str, float]) -> float:
    """
    Map raw column name to S2.2 importance if possible.

    Your S2.2 files often use names like:
      - num__trainspeed
      - cat__equipmenttype_Cut of cars
    while BN evidence uses:
      - 'TrainSpeed_BINS'
      - 'Equipment Type'

    Here we implement a pragmatic mapping:
      1) exact match on raw col
      2) try normalized 'num__<alnumlower>' / 'cat__<alnumlower>_' prefixes
      3) fallback 0
    """
    raw = _normalize_feature_key(col_name)
    if raw in importance_lookup:
        return float(importance_lookup[raw])

    # normalize to alnum lowercase
    base = "".join(ch.lower() for ch in raw if ch.isalnum())
    # try numeric-style name
    num_key = f"num__{base}"
    if num_key in importance_lookup:
        return float(importance_lookup[num_key])

    # try categorical-style: take best match among keys with cat__{base}_
    prefix = f"cat__{base}_"
    best = 0.0
    for k, v in importance_lookup.items():
        if k.startswith(prefix):
            best = max(best, float(v))
    return float(best)


# ----------------------------
# Dataset & preparation
# ----------------------------

@dataclass
class TransformerInputBundle:
    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]
    vocab: Dict[str, int]
    id_to_token: List[str]
    feature_importance_lookup: Dict[str, float]
    meta_df: pd.DataFrame  # aligned with original rows
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray]
    numeric_scaler: Dict[str, np.ndarray]  # {"mean": (F,), "std": (F,)}


class RiskScenarioDataset(Dataset):
    def __init__(
        self,
        token_ids: torch.LongTensor,            # (N, L)
        token_numeric: torch.FloatTensor,       # (N, L, F)
        attention_mask: torch.BoolTensor,       # (N, L) True where token exists
        labels: Optional[torch.FloatTensor] = None,  # (N,)
    ) -> None:
        self.token_ids = token_ids
        self.token_numeric = token_numeric
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "token_ids": self.token_ids[idx],
            "token_numeric": self.token_numeric[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def prepare_transformer_input(
    results_df_bn: pd.DataFrame,
    feature_importances_df: pd.DataFrame,
    max_sequence_length: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    weak_label_type: str = "prob",
    label_transform: str = "none",
    val_ratio: float = 0.0,
    random_seed: int = 42,
    vocab: Optional[Dict[str, int]] = None,
    id_to_token: Optional[List[str]] = None,
) -> TransformerInputBundle:
    """
    Prepare Transformer-ready dataloader from BN results and S2.2 feature importances.

    Each sample is a padded sequence length = max_sequence_length.
      token_ids: (L,)
      token_numeric: (L, 4) => [prob, lift, support_prob, importance]
      attention_mask: (L,) True for real tokens, False for padding
      label: scalar (weak supervision) from prob / lift / uplift (optionally transformed)
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0")

    accident_type_col, evidence_col, prob_col, lift_col, support_col = _infer_bn_columns(results_df_bn)

    df = results_df_bn.copy()
    if evidence_col not in df.columns:
        raise ValueError(f"results_df_bn missing evidence column: expected '{evidence_col}'")

    # standardize support_prob
    if "support_prob" not in df.columns:
        if support_col in df.columns:
            df["support_prob"] = df[support_col].apply(_coerce_float)
        else:
            df["support_prob"] = 0.0

    # standardize prob / lift
    df["prob"] = df[prob_col].apply(_coerce_float) if prob_col in df.columns else 0.0
    df["lift"] = df[lift_col].apply(_coerce_float) if lift_col in df.columns else 0.0
    if "uplift" in df.columns:
        df["uplift"] = df["uplift"].apply(_coerce_float)

    # parse evidence into dict
    df["_evidence_dict"] = df[evidence_col].apply(_safe_parse_evidence)
    df["_accident_type_str"] = df[accident_type_col].astype(str)

    importance_lookup = _build_importance_lookup(feature_importances_df)

    # build token strings per row
    token_seqs: List[List[str]] = []
    token_imps: List[List[float]] = []

    for _, r in df.iterrows():
        acc_type = r["_accident_type_str"]
        ev = r["_evidence_dict"] or {}

        tokens: List[str] = [f"AccidentType={acc_type}"]
        imps: List[float] = [0.0]

        # deterministic order to stabilize training
        for k in sorted(ev.keys(), key=lambda x: str(x)):
            v = ev[k]
            tok = f"{k}={v}"
            tokens.append(tok)
            imps.append(_importance_for_col(str(k), importance_lookup))

        token_seqs.append(tokens)
        token_imps.append(imps)

    # build vocab (or use provided one)
    PAD = "<PAD>"
    UNK = "<UNK>"
    if vocab is None:
        # build new vocab
        vocab = {PAD: 0, UNK: 1}
        for seq in token_seqs:
            for t in seq:
                if t not in vocab:
                    vocab[t] = len(vocab)
        id_to_token = [None] * len(vocab)
        for t, i in vocab.items():
            id_to_token[i] = t
    else:
        # use provided vocab (make a copy to avoid modifying original)
        vocab = dict(vocab)
        # ensure PAD and UNK exist
        if PAD not in vocab:
            vocab[PAD] = 0
        if UNK not in vocab:
            vocab[UNK] = 1
        # rebuild id_to_token from vocab if not provided
        if id_to_token is None:
            max_id = max(vocab.values())
            id_to_token = [None] * (max_id + 1)
            for t, i in vocab.items():
                id_to_token[i] = t
        # else: use provided id_to_token as-is

    # build tensors
    N = len(token_seqs)
    L = max_sequence_length
    F = 4  # prob, lift, support_prob, importance

    token_ids = torch.zeros((N, L), dtype=torch.long)
    token_numeric = torch.zeros((N, L, F), dtype=torch.float32)
    attention_mask = torch.zeros((N, L), dtype=torch.bool)

    for i in range(N):
        seq = token_seqs[i][:L]
        imps = token_imps[i][:L]

        # pad with PAD
        for j in range(L):
            if j < len(seq):
                tid = vocab.get(seq[j], vocab[UNK])
                token_ids[i, j] = tid
                attention_mask[i, j] = True
                token_numeric[i, j, 0] = float(df.iloc[i]["prob"])
                token_numeric[i, j, 1] = float(df.iloc[i]["lift"])
                token_numeric[i, j, 2] = float(df.iloc[i]["support_prob"])
                token_numeric[i, j, 3] = float(imps[j])
            else:
                token_ids[i, j] = vocab[PAD]
                attention_mask[i, j] = False

    # weak labels
    weak_label_type = str(weak_label_type).strip().lower()
    if weak_label_type not in {"prob", "lift", "uplift"}:
        raise ValueError("weak_label_type must be 'prob' or 'lift' or 'uplift'")
    if weak_label_type == "uplift" and "uplift" not in df.columns:
        raise ValueError("weak_label_type='uplift' but results_df_bn has no 'uplift' column")

    labels_np = df[weak_label_type].astype(float).to_numpy()

    label_transform = str(label_transform).strip().lower()
    if label_transform not in {"none", "log1p"}:
        raise ValueError("label_transform must be 'none' or 'log1p'")
    if label_transform == "log1p":
        # for safety: clip to >= -0.999... to avoid log domain error
        labels_np = np.log1p(np.clip(labels_np, -0.999999, None))

    labels = torch.tensor(labels_np, dtype=torch.float32)

    # ---- train/val split (indices) ----
    val_ratio = float(val_ratio)
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")

    rng = np.random.default_rng(int(random_seed))
    all_idx = np.arange(N)
    if val_ratio > 0:
        perm = rng.permutation(all_idx)
        n_val = max(1, int(round(N * val_ratio)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        train_idx = all_idx
        val_idx = None

    # ---- numeric feature standardization (fit on train only) ----
    # token_numeric: (N, L, F) ; use only non-padding tokens from train for stats
    Fdim = token_numeric.shape[-1]
    train_mask = attention_mask[train_idx]  # (Nt, L)
    train_num = token_numeric[train_idx]    # (Nt, L, F)
    # flatten tokens where mask True
    flat = train_num[train_mask].reshape(-1, Fdim) if train_mask.any() else train_num.reshape(-1, Fdim)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)
    # apply to all rows/tokens
    token_numeric = (token_numeric - mean) / std

    dataset = RiskScenarioDataset(token_ids, token_numeric, attention_mask, labels=labels)

    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    if val_idx is not None and len(val_idx) > 0:
        val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        val_loader = None

    meta_df = df.drop(columns=["_evidence_dict", "_accident_type_str"], errors="ignore")

    return TransformerInputBundle(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        vocab=vocab,
        id_to_token=id_to_token,
        feature_importance_lookup=importance_lookup,
        meta_df=meta_df,
        train_indices=train_idx,
        val_indices=val_idx,
        numeric_scaler={"mean": mean.detach().cpu().numpy(), "std": std.detach().cpu().numpy()},
    )


# ----------------------------
# Model
# ----------------------------

class RiskFusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 10,
        numeric_feat_dim: int = 4,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = model_dim
        self.max_sequence_length = max_sequence_length

        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_sequence_length, model_dim)

        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_feat_dim),
            nn.Linear(numeric_feat_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, output_dim),
        )

    def forward(
        self,
        token_ids: torch.LongTensor,        # (B, L)
        token_numeric: torch.FloatTensor,   # (B, L, F)
        attention_mask: torch.BoolTensor,   # (B, L) True for tokens
    ) -> torch.Tensor:
        B, L = token_ids.shape
        device = token_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)

        x = self.token_emb(token_ids) + self.pos_emb(pos) + self.numeric_proj(token_numeric)

        # Transformer expects src_key_padding_mask: True for padding
        src_key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        out = self.head(pooled).squeeze(-1)  # (B,)
        return out


# ----------------------------
# Train / Predict
# ----------------------------

def train_transformer_fusion(
    model: RiskFusionTransformer,
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: str,
    early_stopping_patience: int = 3,
) -> RiskFusionTransformer:
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            token_numeric = batch["token_numeric"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            pred = model(token_ids, token_numeric, attention_mask)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.item()) * token_ids.size(0)
            n += token_ids.size(0)

        train_loss = total_loss / max(n, 1)

        # validation
        val_loss = None
        if val_dataloader is not None:
            model.eval()
            vtot = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    token_ids = batch["token_ids"].to(device)
                    token_numeric = batch["token_numeric"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    y = batch["label"].to(device)
                    pred = model(token_ids, token_numeric, attention_mask)
                    loss = loss_fn(pred, y)
                    vtot += float(loss.item()) * token_ids.size(0)
                    vn += token_ids.size(0)
            val_loss = vtot / max(vn, 1)

            # early stopping check
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

        if val_loss is None:
            logger.info(f"[TransformerFusion] epoch={ep}/{epochs} train_loss={train_loss:.6f}")
        else:
            logger.info(f"[TransformerFusion] epoch={ep}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} (best={best_val:.6f})")

        if val_dataloader is not None and bad_epochs >= int(early_stopping_patience):
            logger.info(f"[TransformerFusion] early stopping triggered (patience={early_stopping_patience})")
            break

    # restore best model if available
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_cri_with_transformer(
    model: RiskFusionTransformer,
    dataloader: DataLoader,
    device: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    preds: List[np.ndarray] = []
    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_numeric = batch["token_numeric"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        p = model(token_ids, token_numeric, attention_mask).detach().cpu().numpy()
        preds.append(p)
    if not preds:
        return np.array([], dtype=float)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_cri_on_full_dataset(
    model: RiskFusionTransformer,
    bundle: TransformerInputBundle,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Predict CRI for all rows in bundle.meta_df in original order.
    (Because train_dataloader / val_dataloader are subsets.)
    """
    # Rebuild a non-shuffled dataloader over the full dataset indices [0..N)
    full_ds: Dataset = bundle.train_dataloader.dataset.dataset  # Subset -> original dataset
    # It is possible train_dataloader is not a Subset in future; guard lightly.
    if not isinstance(full_ds, Dataset):
        raise ValueError("Unexpected dataset type in bundle; cannot reconstruct full dataloader.")
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return predict_cri_with_transformer(model, full_loader, device=device)


