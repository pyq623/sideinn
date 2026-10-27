import math


def sqrt_binary(x, eps=1e-10):
    """有限非负浮点数；区间宽度不超过固定 eps 时停止。"""
    if not math.isfinite(x) or x < 0 or not 0 < eps < 1:
        raise ValueError("x 必须有限非负，eps 必须在 (0, 1) 内")
    if x == 0:
        return 0.0
    # x >= 1 时，sqrt(x) <= x，上界取 x。
    # 0 < x < 1 时，x < sqrt(x) < 1，上界要取 1，不能取 x。
    # 例如 x=0.25，sqrt(x)=0.5；max(1.0, x) 合并了这两种情况。
    left, right = 0.0, max(1.0, x)
    # 区间宽度不超过 eps 就停止，不用整数下标的 left <= right。
    while right - left > eps:
        mid = (left + right) / 2
        if mid == left or mid == right:  # 防止浮点精度极限导致死循环
            break
        if mid < x / mid:  # 相当于 mid * mid < x，除法可避免平方溢出
            left = mid   # 浮点数不能 mid + 1，会跳过答案
        elif mid > x / mid:
            right = mid  # 浮点数不能 mid - 1
        else:
            return mid
    return left  # 返回左端点近似值；区间宽度已满足精度或达到浮点极限


def sqrt_newton(x, eps=1e-10):
    if not math.isfinite(x) or x < 0 or not 0 < eps < 1:
        raise ValueError("x 必须有限非负，eps 必须在 (0, 1) 内")
    if x == 0:
        return 0.0
    y = max(1.0, x)
    for _ in range(2048):
        nxt = 0.5 * y + 0.5 * (x / y)
        # 相邻两次估计值的差不超过固定 eps 就停止。
        if abs(nxt - y) <= eps:
            return nxt
        y = nxt
    raise ArithmeticError("未在迭代上限内收敛")
