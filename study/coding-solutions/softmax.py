import math


def stable_softmax(logits):
    """logits 是一组原始分数；接受有限数或 -inf，全 -inf 时返回全零。"""
    if not logits:
        return []
    if any(math.isnan(score) or score == math.inf for score in logits):
        raise ValueError("不接受 NaN 或 +inf")

    max_score = max(logits)
    # 全部被 mask 时，按本题约定返回全零，避免 -inf - (-inf) 产生 NaN。
    if max_score == -math.inf:
        return [0.0] * len(logits)

    # 每个分数减去相同的最大值，Softmax 结果不变，并能避免指数溢出。
    exp_scores = [math.exp(score - max_score) for score in logits]
    sum_exp_scores = sum(exp_scores)

    # 每项除以指数值之和，得到总和为 1 的概率。
    probabilities = [exp_score / sum_exp_scores for exp_score in exp_scores]
    return probabilities


def stable_softmax_attention(scores):
    """Attention 张量版：scores 为浮点张量 [B, H, Q, K]。

    B 是 batch 大小，H 是 Q 头数，Q/K 是 query/key 序列长度。
    输入允许有限分数和 -inf；此核心版本要求每行至少有一个有限分数。
    """
    import torch

    # 固定 batch、head、query，沿最后一维 K 独立做一次 Softmax。
    # keepdim=True 保留长度为 1 的维度，便于广播回 [B, H, Q, K]。
    max_scores = scores.max(dim=-1, keepdim=True).values  # [B, H, Q, 1]
    exp_scores = torch.exp(scores - max_scores)          # [B, H, Q, K]
    sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True) # [B, H, Q, 1]
    probabilities = exp_scores / sum_exp_scores          # [B, H, Q, K]
    return probabilities
