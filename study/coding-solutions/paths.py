def unique_paths(m, n):
    if m <= 0 or n <= 0:
        return 0
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]


def paths_with_obstacles(grid):
    if not grid or not grid[0]:
        return 0
    dp = [0] * len(grid[0])
    dp[0] = 1
    for row in grid:
        for j, blocked in enumerate(row):
            if blocked:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j - 1]
    return dp[-1]
