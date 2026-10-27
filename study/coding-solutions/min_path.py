def min_path_sum(grid):
    if not grid or not grid[0]:
        return 0
    n = len(grid[0])
    dp = [float("inf")] * n
    dp[0] = 0
    for row in grid:
        for j, value in enumerate(row):
            left = dp[j - 1] if j > 0 else float("inf")
            dp[j] = min(dp[j], left) + value
    return dp[-1]
