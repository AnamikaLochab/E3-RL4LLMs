# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

import numpy as np
import torch

def _canon_index(index, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Return a contiguous int64 tensor of shape [batch_size] on `device`.
    Accepts: list/tuple, np.ndarray (any dtype), or torch.Tensor.
    Maps arbitrary labels (strings, objects) to [0..G-1] via np.unique.
    """
    if torch.is_tensor(index):
        idx_t = index.to(device=device).long().view(-1)
        if idx_t.numel() != batch_size:
            raise ValueError(f"index len {idx_t.numel()} != batch {batch_size}")
        return idx_t

    # Anything non-tensor → NumPy
    arr = np.asarray(index)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != batch_size:
        raise ValueError(f"index len {arr.shape[0]} != batch {batch_size}")

    # If not numeric (or mixed), factorize to ints
    if arr.dtype == np.object_ or not np.issubdtype(arr.dtype, np.integer):
        # Map arbitrary labels to contiguous ints (inverse)
        _, inverse = np.unique(arr.astype(object), return_inverse=True)
        arr = inverse.astype(np.int64, copy=False)
    else:
        arr = arr.astype(np.int64, copy=False)

    return torch.from_numpy(arr).to(device)

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

#change start

def compute_covvar_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    norm_adv: bool = True,
    kappa_clip: float = 1.0,
):
    """
    Cov/Var decorrelated GRPO-style advantage.
    Projects out the component of the group-relative advantage that correlates
    with log-probabilities of the sampled sequences.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        old_log_prob: (bs, response_length)
            log π(y|x) per token. We'll sum over tokens to get sequence logπ.
        index: np.ndarray of group IDs (len = batch_size)
        norm_adv: whether to whiten the corrected advantages
        kappa_clip: clip coefficient for stability

    Returns:
        advantages: (bs, response_length)
        returns:    (bs, response_length) same as advantages for outcome tasks
    """
    scores   = token_level_rewards.sum(dim=-1)              # [bs]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)   # [bs]

    device   = scores.device
    index_np = np.asarray(index)
    corrected_adv = torch.zeros_like(scores)

    for gid in np.unique(index_np):
        mask_np = (index_np == gid)
        mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)

        if mask.sum() < 2:
            # nothing to decorrelate
            scores_g = scores[mask]
            A_g = scores_g - scores_g.mean()
            corrected_adv[mask] = A_g
            continue

        scores_g = scores[mask]        # [G]
        logp_g   = seq_logp[mask]      # [G]

        # --- vanilla GRPO-style group advantage ---
        A_g = scores_g - scores_g.mean()       # baseline
        A_centered    = A_g - A_g.mean()       # often == A_g, defensive
        logp_centered = logp_g - logp_g.mean()

        var_logp = (logp_centered ** 2).mean()
        if var_logp <= 0:
            A_cov = A_g
        else:
            cov   = (A_centered * logp_centered).mean()
            kappa = cov / (var_logp + epsilon)
            kappa = torch.clamp(kappa, -kappa_clip, kappa_clip)
            print("A_centered, Cov, Var, kappa: ", A_centered, cov, var_logp, kappa)

            # decorrelated advantage
            A_cov = A_g - kappa * logp_centered
            print("A_cov: ", A_cov)

        # RLVR-style: only CORRECT sequences get CovVar adjustment,
        # incorrect keep vanilla GRPO advantage.
        correct_mask_local = (scores_g > 0)
        A_cov_final = torch.where(correct_mask_local, A_cov, A_g)

        corrected_adv[mask] = A_cov_final

    # broadcast to tokens
    advantages = corrected_adv.unsqueeze(-1) * response_mask

    if norm_adv:
        advantages = verl_F.masked_whiten(advantages, response_mask)

    print("Final Advantages: ",advantages)
    return advantages, advantages


def compute_covvar_vanilla_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    norm_adv_by_std_in_grpo: bool = True,
    kappa_clip: float = 1.0,):
    """
    Cov/Var decorrelated GRPO-style advantage.

    Matches the structure of compute_grpo_outcome_advantage:
      1) compute per-sequence outcome reward (sum over tokens)
      2) compute group-relative advantage A_i (mean + optional std scaling)
      3) within each prompt group, decorrelate A_i from sequence log-prob
         via a linear projection: A_cov = A - kappa * (logp - mean_logp)
      4) broadcast sequence-level A_cov back to tokens with response_mask

    Args:
        token_level_rewards: (bs, T)
        response_mask:       (bs, T)
        old_log_prob:        (bs, T)  # log π_old per token
        index:               np.ndarray of group IDs, len = bs
        norm_adv_by_std_in_grpo: same meaning as in compute_grpo_outcome_advantage
        kappa_clip:          clip for kappa for stability

    Returns:
        advantages: (bs, T)
        returns:    (bs, T)
    """
    device = token_level_rewards.device
    bs, T = token_level_rewards.shape

    # 1) Outcome reward per sequence (same as GRPO)
    scores = token_level_rewards.sum(dim=-1)  # shape: (bs,)  -> R_i

    # 2) Sequence log-prob under old policy
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)  # shape: (bs,) -> ℓ_i

    # This will store the final sequence-level CovVar-adjusted advantages
    seq_adv_cov = torch.zeros_like(scores, device=device)

    # Build mapping from group id -> list of row indices
    id2indices: dict[int, list[int]] = defaultdict(list)
    for i, idx in enumerate(index):
        id2indices[idx].append(i)

    with torch.no_grad():
        for gid, idx_list in id2indices.items():
            idx_tensor = torch.tensor(idx_list, device=device, dtype=torch.long)

            scores_g = scores[idx_tensor]   # (G,)
            lp_g     = seq_logp[idx_tensor] # (G,)

            G = scores_g.shape[0]
            if G == 1:
                # Same behavior as GRPO: single-sample group -> mean=0, std=1
                if norm_adv_by_std_in_grpo:
                    A_g = scores_g / (1.0 + epsilon)
                else:
                    A_g = scores_g  # scores_g - 0
                seq_adv_cov[idx_tensor] = A_g
                continue

            # ---------- Step 1: GRPO-style group advantage ----------
            if norm_adv_by_std_in_grpo:
                mean_g = scores_g.mean()
                std_g  = scores_g.std(unbiased=False)
                A_g    = (scores_g - mean_g) / (std_g + epsilon)
            else:
                mean_g = scores_g.mean()
                A_g    = scores_g - mean_g
            print("Initial Adv: ",A_g)
            # ---------- Step 2: CovVar decorrelation A ⟂ logp ----------
            A_center  = A_g - A_g.mean()
            lp_center = lp_g - lp_g.mean()

            var_lp = (lp_center ** 2).mean()
            if var_lp <= 0:
                # No variance in log-probs => nothing to decorrelate
                seq_adv_cov[idx_tensor] = A_g
                continue

            cov = (A_center * lp_center).mean()
            kappa = cov / (var_lp + epsilon)
            kappa = torch.clamp(kappa, -kappa_clip, kappa_clip)
            print("Cov, Var, kappa: ",cov, varl_lp, kappa)

            # Decorrelated sequence-level advantage
            A_cov = A_g - kappa * lp_center  # (G,)

            seq_adv_cov[idx_tensor] = A_cov

    # 3) Broadcast sequence-level advantages to tokens and mask
    correct_mask_local = (scores_g > 0)
    A_cov_final = torch.where(correct_mask_local, seq_adv_cov, A_g)
    print("Corrected:", A_cov_final)
    advantages = seq_adv_cov.unsqueeze(-1) * response_mask  # (bs, T)

    # For outcome tasks, returns == advantages (like GRPO)
    return advantages, advantages

def compute_covvar_token_advantage(
    token_level_rewards: torch.Tensor,  # (bs, T)
    response_mask: torch.Tensor,        # (bs, T), 1 for response tokens
    old_log_prob: torch.Tensor,         # (bs, T), log π_old(y_t|h_t)
    index: np.ndarray,                  # (bs,), prompt/group IDs
    epsilon: float = 1e-8,
    norm_adv: bool = True,
    kappa_clip: float = 1.0,
):
    """
    Token-level CovVar GRPO-style advantage.

    - First computes GRPO-style sequence advantages A_i.
    - Broadcasts to tokens to get A_{i,t}^{raw}.
    - Then, per prompt group, decorrelates A_{i,t}^{raw} from token log-probs
      via CovVar at token level.
    - Only correct sequences (R_i > 0) get modified; incorrect tokens keep
      their original GRPO advantage.

    Returns:
        advantages: (bs, T)
        returns:    (bs, T)  (same as advantages for outcome-only tasks)
    """
    device = token_level_rewards.device
    index_t = torch.as_tensor(index, device=device)

    # ----- 1) Sequence reward & group-baseline GRPO advantage -----
    # Sequence reward over response tokens
    seq_reward = (token_level_rewards * response_mask).sum(dim=-1)  # (bs,)

    # Group mean reward per prompt (like GRPO, but without std here)
    id2sum = defaultdict(float)
    id2cnt = defaultdict(int)
    bsz = seq_reward.shape[0]
    for i in range(bsz):
        gid = index[i]
        id2sum[gid] += seq_reward[i].item()
        id2cnt[gid] += 1

    id2mean = {
        gid: torch.tensor(id2sum[gid] / max(id2cnt[gid], 1),
                          device=device, dtype=seq_reward.dtype)
        for gid in id2sum
    }

    adv_seq = seq_reward.clone()
    for i in range(bsz):
        gid = index[i]
        adv_seq[i] = seq_reward[i] - id2mean[gid]

    # ----- 2) Initial token-level advantage (vanilla GRPO broadcast) -----
    # A_{i,t}^{raw} = A_i * response_mask
    A_token_raw = adv_seq.unsqueeze(-1) * response_mask  # (bs, T)

    # We'll build corrected token-level advantages starting from A_token_raw
    corrected = A_token_raw.clone()

    # Identify which sequences are "correct" (RLVR: reward > 0)
    correct_seq = (seq_reward > 0)  # (bs,)

    # ----- 3) Token-level CovVar per prompt group -----
    unique_ids = list(id2mean.keys())
    for gid in unique_ids:
        # sequences belonging to this prompt
        seq_mask = (index_t == gid)              # (bs,)
        if seq_mask.sum() < 2:
            # not enough sequences to estimate covariance
            continue

        # all response tokens in this group
        token_mask_group = seq_mask.unsqueeze(-1) & (response_mask > 0)  # (bs, T)
        if token_mask_group.sum() < 2:
            # no meaningful tokens to work with
            continue

        # Flatten group tokens
        g_group  = A_token_raw[token_mask_group]     # (N_tokens_in_group,)
        lp_group = old_log_prob[token_mask_group]    # (N_tokens_in_group,)

        # Center both
        g_center  = g_group - g_group.mean()
        lp_center = lp_group - lp_group.mean()

        cov = (g_center * lp_center).mean()
        var = (lp_center ** 2).mean()
        kappa = cov / (var + epsilon)

        # Clip κ so we don't go crazy
        kappa = torch.clamp(kappa, 0.0, kappa_clip)

        # If κ is basically zero, nothing to do
        if torch.allclose(kappa, torch.zeros_like(kappa)):
            continue

        # Now apply the correction *only* to tokens from correct sequences
        seq_mask_correct = seq_mask & correct_seq  # (bs,)
        token_mask_correct = seq_mask_correct.unsqueeze(-1) & (response_mask > 0)

        if token_mask_correct.sum() == 0:
            continue

        # Use the group-level lp mean for these tokens
        lp_mean_group = lp_group.mean()

        A_tok_corr = A_token_raw[token_mask_correct]
        lp_corr    = old_log_prob[token_mask_correct]
        lp_center_corr = lp_corr - lp_mean_group

        # CovVar adjustment at token level:
        #   g' = g - κ (logp - mean_logp)
        A_tok_cov = A_tok_corr - kappa * lp_center_corr

        # Write back corrected advantages for correct tokens
        corrected[token_mask_correct] = A_tok_cov

    # ----- 4) Optional whitening over all response tokens -----
    advantages = corrected
    if norm_adv:
        # VERL-style normalization: zero-mean / unit-std on masked tokens
        advantages = verl_F.masked_whiten(advantages, response_mask)

    returns = advantages  # outcome-only tasks

    return advantages, returns

def compute_covvar_token_vanilla_advantage(
    token_level_rewards: torch.Tensor,  # (bs, T)
    response_mask: torch.Tensor,        # (bs, T)
    old_log_prob: torch.Tensor,         # (bs, T) log π_old(y_t)
    index,                              # (bs,) prompt IDs; any hashable type
    epsilon: float = 1e-8,
    use_std: bool = True,               # GRPO vs Dr.GRPO
    kappa_clip: float = 1.0,
):
    """
    CovVar token-level advantage with per-prompt GRPO normalization.
    Returns (advantages, returns) as (bs, T) tensors.
    """
    device = token_level_rewards.device

    # -------------------------------------------------
    # 0) Canonicalize group IDs: arbitrary → {0, ..., G-1}
    # -------------------------------------------------
    index_np = np.asarray(index)
    B = token_level_rewards.size(0)
    assert index_np.shape[0] == B, "index length must match batch size"

    # unique_ids: original group labels; group_ids_int: 0..G-1
    unique_ids, group_ids_int = np.unique(index_np, return_inverse=True)
    group_ids_t = torch.as_tensor(group_ids_int, device=device, dtype=torch.long)  # (B,)

    # -------------------------------------------------
    # 1) Sequence reward R_i
    # -------------------------------------------------
    seq_reward = (token_level_rewards * response_mask).sum(dim=-1)  # (B,)

    # -------------------------------------------------
    # 2) Per-group GRPO normalization in canonical group ID space
    # -------------------------------------------------
    id2scores = defaultdict(list)
    for i, gid_int in enumerate(group_ids_int):      # gid_int: 0..G-1 (numpy int)
        id2scores[int(gid_int)].append(seq_reward[i])

    id2mean, id2std = {}, {}
    for gid_int, vals in id2scores.items():
        vals = torch.stack(vals)                     # all on same device
        id2mean[gid_int] = vals.mean()
        if vals.numel() > 1:
            id2std[gid_int] = vals.std()
        else:
            id2std[gid_int] = torch.tensor(1.0, device=device)

    # GRPO-style sequence-level advantage A_i
    adv_seq = seq_reward.clone()
    for i, gid_int in enumerate(group_ids_int):
        gid_int = int(gid_int)
        if use_std:
            adv_seq[i] = (seq_reward[i] - id2mean[gid_int]) / (id2std[gid_int] + epsilon)
        else:
            adv_seq[i] = seq_reward[i] - id2mean[gid_int]

    # -------------------------------------------------
    # 3) Broadcast to tokens
    # -------------------------------------------------
    response_mask_bool = (response_mask > 0)
    A_token_raw = adv_seq.unsqueeze(-1) * response_mask_bool  # (B, T)

    corrected = A_token_raw.clone()
    correct_seq = (seq_reward > 0)

    # -------------------------------------------------
    # 4) Token-level CovVar per canonical group
    # -------------------------------------------------
    with torch.no_grad():
        for gid_int in id2scores.keys():
            gid_int = int(gid_int)

            # mask over batch dimension for this group → (B,)
            seq_mask = (group_ids_t == gid_int)              # BOOL TENSOR, not Python bool
            # (B, T) mask for all tokens in these sequences
            token_mask_group = seq_mask.unsqueeze(-1) & response_mask_bool

            if token_mask_group.sum() < 2:
                continue

            # flatten group tokens
            g = A_token_raw[token_mask_group]        # (N_tokens,)
            lp = old_log_prob[token_mask_group]      # (N_tokens,)

            g_center  = g - g.mean()
            lp_center = lp - lp.mean()

            cov = (g_center * lp_center).mean()
            var = (lp_center ** 2).mean()
            kappa = torch.clamp(cov / (var + epsilon), 0.0, kappa_clip)
            print("Cov: ", cov.item())
            print("Var: ", var.item())
            print("Kappa: ", kappa.item())

            if kappa.abs() < 1e-9:
                print("kappa<1e-9")
                continue

            # Only correct sequences with positive total reward in this group
            correct_mask = seq_mask & correct_seq               # (B,)
            token_mask_correct = correct_mask.unsqueeze(-1) & response_mask_bool
            if token_mask_correct.sum() == 0:
                continue

            lp_mean = lp.mean()

            A_corr = A_token_raw[token_mask_correct]
            lp_corr = old_log_prob[token_mask_correct]
            lp_center_corr = lp_corr - lp_mean

            # CovVar token-level adjustment
            A_cov = A_corr - kappa * lp_center_corr

            corrected[token_mask_correct] = A_cov

    # -------------------------------------------------
    # 5) Final output
    # -------------------------------------------------
    print("Corrected shape:", corrected.shape)
    # We use corrected both as "advantages" and "returns" here
    return corrected, corrected

def compute_dgrpo_advantage_outcome_A_mod(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs
    tau: float = 0.1,                    # Diversity strength (Correct)
    alpha: float = 0.1,                  # Sharpening strength (Incorrect)
    epsilon: float = 1e-8,
):
    device = token_level_rewards.device
    B = token_level_rewards.size(0)
    idx_t = _canon_index(index, batch_size=B, device=device) 
    # 1. Basic Setup
    scores = token_level_rewards.sum(dim=-1)                  # [B]
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)     # [B]
    
    # 2. Compute Standard GRPO Advantage (Baseline)
    # We do the standard Mean/Std normalization first.
    advantages = torch.zeros_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]
        
        # Standard GRPO Normalization
        if group_scores.numel() > 1:
            mean_g = group_scores.mean()
            std_g = group_scores.std(unbiased=False)
            # Standard Advantage A_i
            advantages[gmask] = (group_scores - mean_g) / (std_g + epsilon)
        else:
            # Fallback for singleton groups
            advantages[gmask] = group_scores - group_scores.mean()

    # 3. Apply Distributional Corrections to A
    # Now we add the diversity/sharpening terms directly to A.
    
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)
        
        # Get data for this group
        group_logp = seq_logp[gidx]
        group_scores = scores[gidx]
        
        # Dynamic Temperature (Crucial for Softmax stability)
        # Using avg_len to scale log-probs to ~[-2, 0] range
        # avg_len = response_mask[gidx].sum(dim=1).float().mean().clamp(min=1.0)
        lengths = (response_mask[gidx] > 0).sum(dim=-1)
        # --- PART A: CORRECT SUBGROUP (Diversity) ---
        corr_mask = (group_scores > 0)
        if corr_mask.sum() > 1:
            # 1. Get Log-Probs of correct items
            valid_idx = gidx[corr_mask]
            log_pi = group_logp[corr_mask]
            lens=lengths[corr_mask]
            # 2. Compute q (Distribution)
            # Scale by temp to prevent Softmax collapse
            log_pi_scaled = log_pi#/lens#avg_len
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()
            
            Hq = -(q * log_q).sum()
            
            # 3. Compute Surprisal
            surprisal = -log_q
            # surprisal = -log_pi_scaled
            # Term = -log q - H(q)
            # If -log q > H(q) (Rare), term is Positive.
            centered_term = surprisal - Hq
            
            # We want to INCREASE advantage for Rare items (High Surprisal)
            # Bonus = tau * (Positive for Rare)
            bonus = tau * centered_term
            # bonus = tau * surprisal
            # Clamp for safety (e.g. max +/- 0.5 change to advantage)
            # bonus = torch.clamp(bonus, -0.5, 0.5)
            print(f"  > [Correct Set] N={corr_mask.sum().item()}")
            print(f"    log p Distribution: {log_pi.detach().cpu().numpy().round(3)}")
            print(f"    q Distribution: {q.detach().cpu().numpy().round(3)}")
            print(f"    Entropy H(q):   {Hq.item():.4f} (Max possible: {np.log(corr_mask.sum().item()):.4f})")
            print(f"    Surprisal (-log q): {surprisal.detach().cpu().numpy().round(2)}")
            print(f"    Centered (S - H):   {centered_term.detach().cpu().numpy().round(2)}")
            print(f"    Final A Bonus:      {bonus.detach().cpu().numpy().round(3)}")
            advantages[valid_idx] += bonus.detach()

        # --- PART B: INCORRECT SUBGROUP (Sharpening) ---
        inc_mask = ~corr_mask
        if inc_mask.sum() > 1:
            bad_idx = gidx[inc_mask]
            log_pi = group_logp[inc_mask]
            lens=lengths[inc_mask]
            # 1. Compute q for the error distribution
            log_pi_scaled = log_pi#/lens #avg_len
            log_q = log_pi_scaled - torch.logsumexp(log_pi_scaled, dim=0)
            log_q = log_q.detach()
            q = log_q.exp()
            
            # 2. Compute Theoretical Entropy H(q)
            Hq = -(q * log_q).sum()
            surprisal = -log_q
            # surprisal = -log_pi_scaled            
            # 3. Center by Entropy
            centered_term = surprisal - Hq
            
            # 4. Subtract from Advantage (Penalize Rare Errors)
            penalty = alpha * centered_term
            # penalty = alpha * surprisal
            # penalty = torch.clamp(penalty, -0.5, 0.5)
            print(f"  > [Correct Set] N={inc_mask.sum().item()}")
            print(f"    log p Distribution: {log_pi.detach().cpu().numpy().round(3)}")
            print(f"    q Distribution: {q.detach().cpu().numpy().round(3)}")
            print(f"    Entropy H(q):   {Hq.item():.4f} (Max possible: {np.log(corr_mask.sum().item()):.4f})")
            print(f"    Surprisal (-log q): {surprisal.detach().cpu().numpy().round(2)}")
            print(f"    Centered (S - H):   {centered_term.detach().cpu().numpy().round(2)}")
            print(f"    Final A penalty:      {penalty.detach().cpu().numpy().round(3)}")
            advantages[bad_idx] -= penalty.detach()
    advantages=advantages.unsqueeze(-1) * response_mask
    return advantages, advantages




def compute_divgrpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    old_log_prob: torch.Tensor,          # [B, T]
    index,                               # group IDs (len B)
    tau: float = 0.5,
    epsilon: float = 1e-8,
    norm_adv_by_std_in_grpo: bool = True,
    diversity_temp: float = None,
):
   
    device = token_level_rewards.device
    B = token_level_rewards.size(0)

    # 1) Canonicalize group IDs
    idx_t = _canon_index(index, batch_size=B, device=device)    # [B], long

    # 2) Sequence-level outcome score and old-policy log prob
    scores   = token_level_rewards.sum(dim=-1)                  # [B]
    total_sequences = scores.numel() # B * G (if flattened)
    total_correct = (scores > 0).sum()
    global_pass_rate = total_correct / total_sequences

    # 2. Set Tau for this step
    current_tau = tau #* global_pass_rate
    seq_logp = (old_log_prob * response_mask).sum(dim=-1)       # [B]

    # 3) GRPO group-wise advantage (mean/std over scores) — baseline we’re modifying
    advantages_seq = torch.empty_like(scores)
    unique_groups = torch.unique(idx_t)

    for gid in unique_groups:
        gmask = (idx_t == gid)
        group_scores = scores[gmask]           # [ng]

        if group_scores.numel() == 1:
            # Degenerate group: no variance. Follow GRPO's convention.
            mean_g = torch.tensor(0.0, device=device)
            std_g  = torch.tensor(1.0, device=device)
            base_adv = group_scores - mean_g
        else:
            mean_g = group_scores.mean()
            if norm_adv_by_std_in_grpo:
                std_g  = group_scores.std(unbiased=False)
                base_adv = (group_scores - mean_g) / (std_g + epsilon)
            else:
                base_adv = group_scores - mean_g

        advantages_seq[gmask] = base_adv

    # This is the "vanilla GRPO" per-sequence advantage if τ = 0
    corrected_adv = advantages_seq.clone()
    print("Original A's:", corrected_adv)
    # 4) dGRPO diversity multiplier per group, only on correct sequences
    for gid in unique_groups:
        gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)   # [ng]

        group_scores = scores[gidx]      # [ng]
        group_logp   = seq_logp[gidx]    # [ng]

        # correct = RLVR-accepted; adjust threshold if you use shaped rewards
        corr = (group_scores > 0)
        if corr.sum() == 0:
            continue

        # --- log π over the full group under old policy ---
        # log_pi = group_logp.detach()                         # [ng], no grad
        lengths = (response_mask[gidx] > 0).sum(dim=-1)
        log_pi = seq_logp[gidx].detach() / lengths
        # log_pi = seq_logp[gidx].detach()
        if diversity_temp is None:
            # Calculate average length of THIS group
            # This acts as a dynamic constant scaler
            avg_len = response_mask[gidx].sum(dim=1).float().mean()
            # Clamp to avoid divide-by-zero or tiny temps
            T = avg_len.clamp(min=1.0) 
        else:
            T = diversity_temp
        # log_pi = log_pi / T

        log_pi_norm = log_pi - torch.logsumexp(log_pi, dim=0)  # log π_g

        # --- conditional distribution over correct indices C ---
        log_q = log_pi_norm[corr]                            # [ncorr]
        log_q = log_q - torch.logsumexp(log_q, dim=0)        # log q_i
        log_q = log_q.detach()

        q  = log_q.exp()                                     # [ncorr]
        print("sum_q:", q.sum()) 
        Hq = (-(q * log_q)).sum().detach()                   # scalar

        # M_i = 1 + τ(-log q_i - H(q)), centered: E_q[ -log q_i - H(q) ] = 0
        mult = (1.0 + current_tau * (-log_q - Hq)).detach()          # [ncorr]
        f = -log_q - Hq
        print(f"\n[dGRPO Monitor] Step Tau: {current_tau:.4f} | Avg Len: {avg_len:.1f}")
        print("f:", f)
        print("E_q[f]:", (q * f).sum())
        print("log q, Hq, (-logq -Hq): ", log_q, Hq , f)
        print("tau, maultiplier: ", current_tau, mult)
        print(f"  > Global Pass Rate: {global_pass_rate:.2%}")
        # Optional: clip for stability, e.g.:
        # mult = mult.clamp(0.5, 2.0)

        # apply only to correct samples in this group
        tgt = gidx[corr]
        corrected_adv[tgt] = corrected_adv[tgt] * mult
    print("Updated A's:", corrected_adv)
    # 5) Broadcast sequence-level advantages to tokens, mask by response
    advantages = corrected_adv.unsqueeze(-1) * response_mask  # [B, T]

    # IMPORTANT: no extra whitening here; we already did GRPO-style group norm
    return advantages, advantages

# def compute_divgrpo_outcome_advantage(
#     token_level_rewards: torch.Tensor,
#     response_mask: torch.Tensor,
#     old_log_prob: torch.Tensor,
#     index,                        # can be anything; we’ll canonify it
#     tau: float = 0.5,
#     epsilon: float = 1e-8,
#     norm_adv: bool = True,
# ):
#     device = token_level_rewards.device
#     B = token_level_rewards.size(0)
#     print("tau:",tau)
#     # ✅ robust, contiguous, device-correct group IDs
#     idx_t = _canon_index(index, batch_size=B, device=device)    # [B], long

#     # outcome-only score per sequence
#     scores   = token_level_rewards.sum(dim=-1)                  # [B]
#     seq_logp = (old_log_prob * response_mask).sum(dim=-1)      # [B] (from old policy)

#     # group baseline (mean) like GRPO
#     advantages_seq = scores.clone()
#     for gid in torch.unique(idx_t):
#         gsel = (idx_t == gid)
#         gmean = scores[gsel].mean()
#         advantages_seq[gsel] -= gmean

#     # dGRPO diversity multiplier inside correct set
#     corrected_adv = advantages_seq.clone()
#     for gid in torch.unique(idx_t):
#         gidx = torch.nonzero(idx_t == gid, as_tuple=False).squeeze(-1)   # [ng]

#         # log pi over the group from old policy (no grad path)
#         log_pi = seq_logp[gidx]                                          # [ng]
#         log_pi_norm = log_pi - torch.logsumexp(log_pi, dim=0)            # log pi_g

#         # restrict to "correct"; adapt the threshold if your reward isn't binary
#         corr = scores[gidx] > 0
#         if corr.sum() == 0:
#             continue

#         # IMPORTANT: conditional renormalization **within correct only**
#         log_q = log_pi_norm[corr]
#         log_q = log_q - torch.logsumexp(log_q, dim=0)                    # log q(y|correct)

#         # Detach: treat multipliers as constants for PPO/GRPO update
#         log_q = log_q.detach()
#         q     = log_q.exp()
#         Hq    = (-(q * log_q)).sum().detach()                             # scalar
#         mult  = (1 + tau * (-log_q - Hq)).detach()                        # [ncorr]
#         print(mult)
#         # mult = mult.clamp(1 - clip, 1 + clip)

#         # apply only to correct samples in this group
#         tgt = gidx[corr]
#         corrected_adv[tgt] = corrected_adv[tgt] * mult

#     advantages = corrected_adv.unsqueeze(-1) * response_mask
#     if norm_adv:
#         advantages = verl_F.masked_whiten(advantages, response_mask)

#     return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


#change end

def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
