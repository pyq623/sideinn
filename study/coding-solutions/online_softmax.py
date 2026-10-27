import math


def online_softmax(logits):
    """logits 是一组原始分数；两遍计算，全 -inf 时返回全零。"""
    if not logits:
        return []

    # 第一遍：逐个读入分数，维护当前最大值和对应的指数和。
    # 关键：sum_exp_scores 是已处理分数减去当前 max_score 后的指数和，
    # 即 sum(exp(已处理分数 - max_score))，不是原始的 sum(exp(已处理分数))。
    max_score = -math.inf
    sum_exp_scores = 0.0
    for score in logits:
        if math.isnan(score) or score == math.inf:
            raise ValueError("不接受 NaN 或 +inf")
        # 被 mask 的分数不贡献概率；跳过也能避免 -inf - (-inf)。
        if score == -math.inf:
            continue

        new_max_score = max(max_score, score)
        # exp(旧分数 - 新最大值) = exp(旧分数 - 旧最大值) * exp(旧最大值 - 新最大值)。
        # 最大值变大，旧项应该缩小，所以乘 exp(max_score - new_max_score)。
        # 新分数也要减去 new_max_score，不能直接加 exp(score)。
        sum_exp_scores = (
            sum_exp_scores * math.exp(max_score - new_max_score)
            + math.exp(score - new_max_score)
        )
        max_score = new_max_score

    # 全部被 mask 时，按本题约定返回全零。
    if max_score == -math.inf:
        return [0.0] * len(logits)

    # 第二遍：用最终的最大值和指数和，计算每个分数对应的概率。
    probabilities = [math.exp(score - max_score) / sum_exp_scores for score in logits]
    return probabilities
