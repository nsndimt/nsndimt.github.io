from collections import deque
from itertools import accumulate
from typing import List


class PrefixSum:
    def __init__(self, arr: List[int]):
        self.prefixsum = list(accumulate(arr, initial=0))

    def query(self, i: int, j: int) -> int:
        # 查询的是双闭区间[i, j]的区间和
        return self.prefixsum[j + 1] - self.prefixsum[i]


class MatrixPrefixSum:
    def __init__(self, mat: List[List[int]]):
        self.n, self.m = len(mat), len(mat[0])
        self.prefixsum = [x.copy() for x in [[0] * (self.n + 1)] * (self.m + 1)]

        for i in range(self.m):
            for j in range(self.n):
                self.prefixsum[i + 1][j + 1] = (
                    -self.prefixsum[i][j] + self.prefixsum[i][j + 1] + self.prefixsum[i + 1][j] + mat[i][j]
                )

    def query(self, row1, col1, row2, col2):
        return (
            -self.prefixsum[row1][col2 + 1]
            - self.prefixsum[row2 + 1][col1]
            + self.prefixsum[row2 + 1][col2 + 1]
            + self.prefixsum[row1][col1]
        )


class DifferentialArray:
    def __init__(self, arr: List[int]):
        self.diff = [arr[0]] + [arr[i] - arr[i - 1] for i in range(1, len(arr))] + [-arr[i]]

    # 取[i~j]的双闭区间进行区间修改
    def modify(self, i: int, j: int, value: int):
        self.diff[i] += value  # 复原时, arr[i]之后的数都会 + value
        self.diff[j + 1] -= value  # 抵消arr[i]修改

    # 一连串的modify最后recover 返回修改后的数组
    def recover(self) -> List[int]:
        return list(accumulate(self.diff))[:-1]


class MatrixDifferentialArray:
    def __init__(self, mat: List[List[int]]):
        self.n, self.m = len(mat), len(mat[0])
        # 下标从0开始
        # 最后一行无用 最后一列无用
        self.diff = [x[:] for x in [[0] * (self.n + 1)] * (self.m + 1)]
        for i in range(self.m):
            for j in range(self.n):
                self.insert(i, i, j, j, mat[i][j])

    def insert(self, r1: int, c1: int, r2: int, c2: int, v: int):
        self.diff[r1][c1] += v
        self.diff[r1][c2 + 1] -= v
        self.diff[r2 + 1][c1] -= v
        self.diff[r2 + 1][c2 + 1] += v

    def recover(self) -> List[List[int]]:
        ans = [x.copy() for x in [[0] * (self.n + 1)] * (self.m + 1)]
        for i in range(self.m):
            for j in range(self.n):
                ans[i + 1][j + 1] = ans[i][j + 1] + ans[i + 1][j] + diff[i][j] - ans[i][j]
        return [ans[i + 1][1:] for i in range(self.m)]

arr = [1, 2, 4, 6, 3, 5]
# 正向遍历
s = deque()
previous_greater = [-1] * len(arr)
next_greater_or_equal = [len(arr)] * len(arr)
for i, x in enumerate(arr):
    # 当前元素更大更近 => 比当前元素小出栈
    # 比当前元素小出栈 => 找到了下一个更大
    while len(s) > 0 and arr[s[-1]] <= x:
        next_greater_or_equal[s.pop()] = i
    previous_greater[i] = -1 if len(s) == 0 else s[-1]
    s.append(i)

s = deque()
previous_less = [-1] * len(arr)
next_less_or_equal = [len(arr)] * len(arr)
for i, x in enumerate(arr):
    # 当前元素更小更近 => 比当前元素大出栈
    # 比当前元素大出栈 => 找到了下一个更小
    while len(s) > 0 and arr[s[-1]] >= x:
        next_less_or_equal[s.pop()] = i
    previous_less[i] = -1 if len(s) == 0 else s[-1]
    s.append(i)

# 反向遍历
s = deque()
previous_greater_or_equal = [-1] * len(arr)
next_greater = [len(arr)] * len(arr)
for i, x in list(enumerate(arr))[::-1]:
    while len(s) > 0 and arr[s[-1]] <= x:
        previous_greater_or_equal[s.pop()] = i
    next_greater[i] = -1 if len(s) == 0 else s[-1]
    s.append(i)

s = deque()
previous_less_or_equal = [-1] * len(arr)
next_less = [len(arr)] * len(arr)
for i, x in list(enumerate(arr))[::-1]:
    while len(s) > 0 and arr[s[-1]] >= x:
        previous_less_or_equal[s.pop()] = i
    next_less[i] = -1 if len(s) == 0 else s[-1]
    s.append(i)
