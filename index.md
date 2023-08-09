---
title: LeetCode Note for nsndimt
layout: default
---

* TOC
{:toc}

# Python语言
- `dims=[state_dim, *dims, action_dim]`构造数组而不是`dims=[state_dim] + dims + [action_dim]`
- 二维数组创建和下标访问就用`[x[:] for x in [[0]*1000]*1000]`除非有向量化运算才用`numpy`
- `a, b = a + b, a - b` 的计算顺序为：
	- `a + b`
	- `a - b`
	- 赋值 a
	- 赋值 b
- `a = b = 1 + 1` 的计算顺序为：
	- `1 + 1`
	- 赋值 a
	- 赋值 b
 - if `i < len(arr) and arr[i] > 0` 的计算顺序为：
     -  `i < len(arr)``
     -  `arr[i] > 0` 如果第一个条件为假则不计算(short circuiting)
- `a <= b <= c` 的计算顺序为：
	- `a <= b`
	- `b <= c`
- `max(iterable, *, default=obj, key=None)` 和 `max(iterable, *, default=obj, key=None)`
    - default指定iterable为空时返回值
    - key指定比较值
- `min` and `argmin` 一次性拿到 `min_v, min_idx = min(enumerate(x), key=operator.itemgetter(1)` 
- `enumerate(iterable, start=0)` 用于 `enumerate(arr[1:], start=1)`
- `defaultdict(list)` 空数组 `defaultdict(int)` 默认为0 `defaultdict(lambda:-1)` 默认为-1
- `{'jack',}` 等价 `set('jack')` 以及 `{'jack':1}` 等价 `dict(jack=1)`
- python 整除取余向负无穷取整 `n mod base = n - math.floor(n/base) * base` 要模仿C的整除取余行为(向零取整)用int() `n - int(n/base) * base`
    - `7//4 = 1` 和 `7%4 = 3`
    - `-7//4 = -2` 和 `-7%4 = 1`
    - `7//-4 = -2` 和 `7%-4 = -1`
    - `-7//-4 = 1` 和 `-7%-4 = -3`
- 字符串整数/进制转换
    - `int(x, base=10)` `str("0x21")`
    - 十进制: `digits = list(map(int, str(x)))`
    - 二进制 0b前缀: `digits = [int(x) for x in bin(x)[2:]]`
    - 八进制 0o前缀: `digits = [int(x) for x in oct(x)[2:]]`
    - 十六进制 0x前缀: `digits = [int(x) for x in hex(x)[2:]]`
- 活用itertools
    - `product('ABCD', repeat=2)` --> AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
    - `permutations('ABCD', 2)` --> AB AC AD BA BC BD CA CB CD DA DB DC
    - `combinations('ABCD', 2)` -->AB AC AD BC BD CD
    - `combinations_with_replacement('ABCD', 2)` --> AA AB AC AD BB BC BD CC CD DD
    - `islice(iterable, start, stop[, step])` `arr[start:end:step]` O(end-start)
    - `groupby(iterable, key=None)`
        - ``[k for k, g in groupby('AAAABBBCCDAABBB')]`` --> A B C D A B
        - ``[list(g) for k, g in groupby('AAAABBBCCD')]`` --> AAAA BBB CC D
- nonlocal声明只在当对函数外部mutable变量赋值时必须: 在Python 2中，闭包只能读外部函数的变量，而不能改写它。为了解决这个问题，Python 3引入了nonlocal关键字，在闭包内用nonlocal声明变量，就可以让解释器在外层函数中查找变量名。
- Python里只有2种作用域：全局作用域和局部作用域。全局作用域是指当前代码所在模块的作用域，局部作用域是指当前函数或方法所在的作用域。局部作用域里的代码可以读包括全局作用域里的变量，但不能更改它。如果想更改它，这里就要使用global关键字了

# 排序
- 大部分有序排序
	- 1、最理想情况（数据预先已排好序），插入和冒泡都只需要进行n次循环和比较就结束了，不需要进行数据交换（传值），而选择要进行(n^2)/2次循环和比较，显然选择明显落后于冒泡和插入
	- 2、平均情况（数据完全随机），插入要进行(n^2)/4次循环和比较，以及同是(n^2)/4次的数据复制（传值）而冒泡要进行(n^2)/2次循环和比较，(n^2)/4次交换，每次交换等于3次数据复制（传值），因此它的循环比较次数和和数据复制次数分别是插入的2倍和3倍，因此冒泡的耗时是插入的2-3倍之间
 	- 如果排序对象是简单数据，每次比较操作的耗时大于数据复制，则冒泡的耗时会比较接近插入的2倍；如果排序对象是很长的数据结构，每次复制数据的耗时远大于比较，则冒泡的耗时会更接近插入的3倍。
- 前K大 1. 冒泡 K 遍 2. 维护大小为K最大堆 3.递归划分（快排 quickselect）
- 逆序对: 归并排序

```python
def bubblesort(arr):
    swapped = False
    for n in range(len(arr) - 1, 0, -1):
        for i in range(n):
            if arr[i] > arr[i + 1]:
                swapped = True
                arr[i], arr[i + 1] = arr[i + 1], arr[i]       
        if not swapped:
            return

def selectionSort(arr):
    for idx in range(len(arr)):
        min_v, min_idx = arr[idx], ind
        for j in range(idx + 1, len(arr)):
            if array[j] < min_v:
                min_v, min_idx = array[j], j
        arr[idx], arr[min_idx] = arr[min_idx], array[idx]

def insertionSort(arr):
    for i, key in enumerate(arr):
        prev = i - 1
        while j >= 0 and arr[j] > key:
            arr[prev + 1] = arr[prev]
            j -= 1
        arr[prev + 1] = key

def mergeSort(arr, left, right, output=None):
    if output is None:
        output = [0] * len(arr)

    inv_count = 0
    if left < right:
        mid = (left + right)//2
        inv_count += mergeSort(arr, left, mid, output)
        inv_count += mergeSort(arr, mid + 1, right, output)

        i = left     # Starting index of left subarray
        j = mid + 1  # Starting index of right subarray
        k = left     # Starting index of to be sorted subarray

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                output[k] = arr[i]
                k += 1
                i += 1
            else:
                output[k] = arr[j]
                inv_count += mid - i + 1
                k += 1
                j += 1

        while i <= mid:
            output[k] = arr[i]
            k += 1
            i += 1

        while j <= right:
            output[k] = arr[j]
            k += 1
            j += 1

        for p in range(left, right + 1):
            arr[p] = output[p]
 
    return inv_count

# Lomuto partition(slow but easy)
# all elements with values less or equal to the pivot come before the pivot
# all elements with values greater than the pivot come after it
# return exact pivot

def partition(arr, low, high):
    # care about where pivot is
    pivot = arr[high]

    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:             
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Hoare partition
# all elements with values less or equal to the pivot come before the pivot
# all elements with values greater or equal to the pivot come after it
# dose not return exact pivot
def partition(arr, low, high):
    # not care about where pivot is
    pivot = arr[high]

    i = low - 1
    j = high + 1
 
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1

	    j -= 1
        while arr[j] > pivot:
            j -= 1

        if i >= j:
            return j
 
        arr[i], arr[j] = arr[j], arr[i]

# for Lomuto partition
# when adapted for Hoare partition, only choose median of three(low, high, median) 
def medianofthree(arr, low, high):
    mid = (low + high) // 2
    if arr[mid] < arr[low]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[high] < arr[low]
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] < arr[high]
        arr[low], arr[mid] = arr[mid], arr[low]
    pivot = arr[high]

def quickSort(arr, low, high):
    if (low < high):
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)

def topKlargest(arr, k):
    left = 0
    right = len(arr)-1
    while True:
        # must be lomuto since we need idx > p elements to be strictly greater than pivot (sorted)
        # and idx = p equal to pivot, there might be more element equal to pivot, but at least idx >= p must be top len(arr) - p
        p = LomutoPartition(arr, left, right)
        if p == len(arr) - k:
            return [t[1] for t in arr[p:]]
        elif p < len(arr) - k:
            left = p + 1
        else:
            right = p - 1
```

- `cmp_to_key` 定义复杂顺序 `operator.itemgetter`替代lambda

```python
def cmp(a, b):
    if a[0] > b[0]:
        return 1
    elif a[0] < b[0]:
        return -1
    else:
        return 0

sorted([(1, 2), (4, 2)], key=functools.cmp_to_key(cmp))
```

- 自定义`heapq`和`SortedContainer`比较函数

```python
class Node:
    def __init__(self, val: int):
        self.val = val
    # 有eq才是全序没有偏序
    def __eq__(self, other):
        return self.val == other.val

	def __lt__(self, other):
        return self.val < other.val
    # lt，gt两者有一个即可 a lt b == b gt a
    def __gt__(self, other):
        return self.val > other.val

heap = [Node(2), Node(0), Node(1), Node(4), Node(2)]
heapq.heapify(heap)
print(heap)  # output: [Node value: 0, Node value: 2, Node value: 1, Node value: 4, Node value: 2]
```

# 前缀和 差分

```python 
prefixsum = [0] + list(accumulate(arr))
def query(i, j):
    # 查询的是双闭区间[i, j]的区间和
    return prefixsum[j+1] - prefixsum[i]

# 动态建立前缀和
def subarraySum(self, nums: List[int], k: int) -> int:
    n = len(nums)
    s = 0
    ans = 0
    prefix_sum_cnt = Counter([0])
    for num in nums:
        s = s + num
        ans += prefix_sum_cnt[s - k]
        prefix_sum_cnt[s] += 1
    return ans

suffixsum = list(accumulate(arr)) + [0]
def query(i, j):
    # 查询的是双闭区间[i, j]的区间和
    return suffixsum[i] - suffixsum[j+1]

m = len(matrix)
n = len(matrix[0])
prefix_sum = prefix_sum = [x[:] for x in [[0]*(n+1)]*(m+1)]
for i in range(m):
    for j in range(n):
        prefix_sum[i+1][j+1] = prefix_sum[i][j+1] + prefix_sum[i+1][j] + matrix[i][j] - prefix_sum[i][j]

def query(row1, col1, row2, col2):
    return - prefix_sum[row1][col2+1] - prefix_sum[row2+1][col1] + prefix_sum[row2+1][col2+1] + prefix_sum[row1][col1]

diff = [arr[0]] + [arr[i] - arr[i - 1] for i in range(1, len(arr))]
def modify(i, j, value):
    # 取[i~j]的双闭区间进行区间修改
    diff[i] += value  # 复原时, arr[i]之后的数都会 + value
    if j + 1 < len(diff):
        diff[j + 1] -= value
# 一连串的modify最后recover
def recover():
    # 复原修改后的数组
    res = [diff[0]]
    for i in range(1, len(diff)):
        res.append(res[-1] + diff[i])
    return res

def checkArray(self, nums: List[int], k: int) -> bool:
    n = len(nums)
    d = [0] * (n + 1)
    sum_d = 0
    for i, x in enumerate(nums):
        sum_d += d[i]
        x += sum_d
        if x == 0: continue  # 无需操作
        if x < 0 or i + k > n: return False  # 无法操作
        sum_d -= x  # 直接加到 sum_d 中
        d[i + k] += x
    return True

```
# 树状数组
![image](/assets/array_tree.png)

```python
# lowbit: x 的二进制中，最低位的 1 以及后面所有 0 组成的数
# -x: 补码定义 最低1左侧取反，右侧不变
# x & -x

class BinaryIndexTree:

    def __init__(self, nums: List[int]):
        # idx starts at 1 so need to pad 0 at start
        prefix_sum = list(accumulate([0]+nums))
        n = self.n = len(nums)
        self.arr = [0] * (n + 1)
        for i in range(1, n + 1):
            self.arr[i] = prefix_sum[i] - prefix_sum[i - (i & (-i))]
        self.nums = nums

    def update(self, index: int, val: int) -> None:
        diff = val - self.nums[index]
        self.nums[index] = val
        idx = index + 1
        while idx <= self.n: # 不能越界
            self.arr[idx] += diff
            idx += idx & -idx

    def prefix_sum(self, x): # a[1]..a[x]的和
        ans = 0
        while x > 0:
            ans = ans + self.arr[x]
            x = x - (x & -x)
        return ans
    
    def sumRange(self, left: int, right: int) -> int:
        return self.prefix_sum(right+1) - self.prefix_sum(left)
```

# 双指针 滑动窗口

- 寻找单调性进行优化O(N^2) 循环
  - 同向双指针 纯靠模拟 拿出笔来 注意边界条件 子数组 子序列
  - 对向双指针
- 快慢指针
- 分组循环

```python
#同向双指针 求最小窗口 小区间满足包含它的大区间一定满足
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    n = len(nums)
    ans = n + 1
    s = 0
    left = 0
    for right in range(n):
        s += nums[right]
        # when left = right, s must be 0 and the target always > 0
        # left <= right is not needed in this case but is necessary in other cases 
        #窗口合法 → 开始收缩
        while left <= right and s >= target:
            # [left, right] has sum >= target
            # but not every valid window ended with right can enter the loop body
            ans = min(right - left + 1, ans)
            s -= nums[left]
            left += 1
        # when subarray is empty left = right + 1
    return ans if ans <= n else 0
#同向双指针 求最大窗口 大区间满足子区间一定满足
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    n = len(nums)
    p = 1
    ans = 0
    left = 0
    for right in range(n):
        p *= nums[right]
        #窗口非法 → 开始收缩
        while left <= right:
            if p < k:
                # have found every valid largest window ended with right
                ans += right - left + 1
                break
            p //= nums[left]
            left += 1
    return ans

def alternatingSubarray(self, nums: List[int]) -> int:
    n = len(nums)
    start = 0
    ans = -1
    while start < n - 1:
        # 寻找开头
        if nums[start + 1] - nums[start] != 1:
            start += 1
        else:
            end = start + 1
            # 枚举合法下一个
            while end + 1 < n and nums[end + 1] == nums[start] + (end + 1 - start) % 2:
                end += 1
            ans = max(ans, end - start + 1)
            # 最多回退2 3434545 345
            start = end
    return ans
```

# 二分

```python
def lastlessequal(arr, v):
    left = -1
    right = len(arr)
    while left + 1 < right:
        mid = left + (right - left) // 2
        if arr[mid] <= v:
            left = mid
        else:
            right = mid
    return left

def lastlessequal(arr, v):
    left = 0
    right = len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= v:
            left = mid + 1
        else:
            right = mid
    return left

def lastlessequal(arr, v):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] <= v:
            left = mid + 1
        else:
            right = mid - 1
    return left

def lastlessequal(arr, v):
    left = -1
    right = len(arr)
    while left < right:
        mid = left + (right - left+ 1) // 2
        if arr[mid] <= v:
            left = mid
        else:
            right = mid - 1
    return left

def equal(arr, v):
    left = -1
    right = len(arr)
    while left + 1 < right:
        mid = left + (right - left) // 2
        if arr[mid] < v:
            left = mid
        elif arr[mid] == v:
            return mid
        else:
            right = mid
    return -1
```

- 核心要素: 区间染色 红，蓝，未知
    - 注意区间开闭，三种都可以
    - 循环结束条件：当前区间内没有元素
    - 下一次二分查找区间：不能再查找(区间不包含)mid，防止死循环 => 区间恒缩小
    - 判断条件，返回值：取决于寻找什么 和区间开闭无关

## 二分和bisect关系  

| position            | bisect                          | minium possible value | maxium possible value |
| -----------         | -----------                     | -----------           | -----------           |
| last less or equal  | bisect.bisect_right(arr, v) - 1 | -1 (not find)         | len(arr) - 1          |
| first great         | bisect.bisect_right(arr, v)     | -1                    | len(arr) (not find)   |
| last less           | bisect.bisect_left(arr, v) - 1  | -1 (not find)         | len(arr) - 1          |
| first great or equal| bisect.bisect_right(arr, v)     | -1                    | len(arr) (not find)   |

## 循环不变量

| position            | if condition    | return                       | red         | blue        |
| -----------         | -----------     | -----------                  | ----------- | ----------- |
| last less or equal  | arr[mid] <= v   | left                         | <= v        | > v         |
| first great         | arr[mid] <= v   | right                        | <= v        | > v         |
| last less           | arr[mid] < v    | left                         | < v         | >= v        |
| first great or equal| arr[mid] < v    | right                        | < v         | >=v         |
| equal               | arr[mid] <> v   | -1 all colored => not found  | < v         | >=v         |

## 开区间
- 因为未检查区间 `(left, right)` 为开区间 所以`left = -1; right = len(arr)`
- 红区间: `[0, left]` 蓝区间: `[right, len(arr) - 1]`
- 因为未检查区间 `(left, right)` 为开区间 所以`left + 1 == right`时区间为空 `while left + 1 < right:`
- `left = -1; right = 1` => `mid = (left + right) // 2`
- 因为未检查区间 `(left, right)` 为开区间 所以 `left = mid`和`right = mid`不会导致mid留在未检查区间里

## 左闭右开区间
- 因为未检查区间 `[left, right)` 为左闭右开区间 所以`left = 0; right = len(arr)`
- 红区间: `[0, left)` 蓝区间: `[right, len(arr) - 1]`
- 因为未检查区间 `[left, right)` 为左闭右开区间 所以`left == right`时区间为空 `while left < right:`
- `left = 0; right = 1` => `mid = (left + right) // 2`
- 因为未检查区间 `[left, right)` 为左闭右开区间 所以 `left = mid + 1`和`right = mid`不会导致mid留在未检查区间里

## 闭区间
- 因为未检查区间 `[left, right]` 为闭区间 所以`left = 0; right = len(arr) - 1`
- 红区间: `[0, left)` 蓝区间: `(right, len(arr) - 1]`
- 因为未检查区间 `[left, right]` 为闭区间 所以`left == right + 1`时区间为空 `while left <= right:`
- `left = 0; right = 0` => `mid = (left + right) // 2`
- 因为未检查区间 `[left, right]` 为闭区间 所以 `left = mid + 1`和`right = mid - 1`不会导致mid留在未检查区间里

## 左开右闭区间
- 因为未检查区间 `(left, right]` 为左开右闭区间 所以`left = -1; right = len(arr) - 1`
- 红区间: `[0, left]` 蓝区间: `(right, len(arr) - 1]`
- 因为未检查区间 `(left, right]` 为左开右闭区间 所以`left == right`时区间为空 `while left < right:`
- `left = -1; right = 0` => `mid = (left + right + 1) // 2`
- 因为未检查区间 `(left, right]` 为左开右闭区间 所以 `left = mid`和`right = mid - 1`不会导致mid留在未检查区间里

# 递归

## BFS

```python
# 二维迷宫BFS
# starts: 起点x, y坐标
# exits: 终点x, y坐标
# grid: 迷宫 0代表可行 1代表障碍

def BFS(starts, exits, grid):
    m, n = len(maze), len(maze[0])
    q = deque()
    vis = [row.copy() for row in [[False] * n]*m]
    end = [row.copy() for row in [[False] * n]*m]
    for x, y in starts:
        q.append((x, y, 0))
    for x, y in exits:
        end[x][y] = True

    while q:
        i, j, d = q.popleft()
        for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
            if 0 <= x < m and 0 <= y < n and grid[x][y] == 0 and not vis[x][y]:
                q.append((x, y, d+1))
                vis[x][y] = True
                if end[x][y]:
                    return d+1
    return -1

# version 2 处理一次多步 所以一个点可能被反复访问只要距离更近
def BFS(starts, exits, grid):
    m, n = len(maze), len(maze[0])
    q = []
    dis = [row.copy() for row in [[1<<31] * n]*m]
    end = [row.copy() for row in [[False] * n]*m]
    for x, y in starts:
        q.append((x, y, 0))
    for x, y in exits:
        end[x][y] = True

    while q:
        qfreeze = q
        q = []
        for i, j，d in qfreeze:
            for x, y in (i + 1, j), (i + 1, j + 1), (i, j + 1), (i - 1, j + 1), (i - 1, j), (i - 1, j - 1), (i, j - 1), (i + 1, j - 1):
                # dis 兼做判断是否已经访问过
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 0 and dis[x][y] > d + 1:
                    q.append((x, y, d + 1))
                    dis[x][y] = d + 1
                    if end[x][y]:
                        return d+1
    return -1
```
## DFS

### 子集型

```python
# 单个元素的视角 选或不选
def subsets(self, nums: List[int]) -> List[List[int]]:
    ans = []
    path = []
    n = len(nums)
    def dfs(i: int) -> None:
        if i == n:
            ans.append(path.copy())  # 固定答案
            return
        # 不选 nums[i]
        dfs(i + 1)
        # 选 nums[i]
        path.append(nums[i])
        dfs(i + 1)
        path.pop()  # 恢复现场
    dfs(0)
    return ans
# 答案的视角 下一个是谁
def subsets(self, nums: List[int]) -> List[List[int]]:
    ans = []
    path = []
    n = len(nums)
    def dfs(i: int) -> None:
        ans.append(path.copy())  # 固定答案
        if i == n:
            return
        for j in range(i, n):  # 枚举选择的数字
            path.append(nums[j])
            dfs(j + 1)
            path.pop()  # 恢复现场
    dfs(0)
    return ans
```

### 组合型

```python
# 单个元素的视角 选或不选
def combine(self, nums: List[int], k: int) -> List[List[int]]:
    ans = []
    path = []
    n = len(nums)
    def dfs(i: int) -> None:
        d = k - len(path)  # 还要选 d 个数
        if d == 0:
            ans.append(path.copy())
            return
        # 不选 i
        if n - i > d:
            dfs(i + 1)
        # i时包括i只剩n-i所以小于等于d时必选
        # 选 i
        path.append(nums[i])
        dfs(i + 1)
        path.pop()
    dfs(0)
    return ans

# 答案的视角 下一个是谁
def combine(self, nums: List[int], k: int) -> List[List[int]]:
    ans = []
    path = []
    n = len(nums)
    def dfs(i: int) -> None:
        d = k - len(path)  # 还要选 d 个数
        if d == 0:
            ans.append(path.copy())
            return
        # 还要选d个意味着下一个不可能比n-d大
        for j in range(i, n-d+1):
            path.append(j)
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans
```

### 排列

```python
def permute(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    ans = []
    path = []
    on_path = [False] * n
    def dfs(i: int) -> None:
        if i == n:
            ans.append(path.copy())
            return
        for j in range(n):
            if not on_path[j]:
                path.append(nums[j])
                on_path[j] = True
                dfs(i + 1)
                on_path[j] = False  # 恢复现场
                path.pop()
    dfs(0)
    return ans
```

# 数学

## 数论

```python
def gcd(x, y):
    while y != 0:
        x, y = y, x % y
    return x

def binpow(a, b):
    res = 1
    while b > 0:
        if (b & 1):
            res = res * a
        a = a * a
        b >>= 1
    return res

math.comb(x, y)
math.perm(x, y)
math.factorial(n)
math.isqrt(x) == math.floor(math.sqrt(x))
math.gcd(x, y)
math.lcm(x, y)

maxab = lambda a, b: a if a > b else b
minab = lambda a, b: a if a < b else b
abs = n if n > 0 else -n
isodd = (n & 1 == 1) = (n % 2 == 1)

aseert n^n == 0
assert -n == ~n + 1

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    for x in range(3, math.isqrt(n) + 1, 2):
        if number % x == 0:
            return False
    return True


def breakdown(n):
    result = []
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0: # 如果 i 能够整除 N，说明 i 为 n 的一个质因子。
            while n % i == 0:
                n //= i
            result.append(i)
    if n != 1: # 说明再经过操作之后 n 留下了一个素数
        result.append(n)
    return result

def prime(n):
    flag = [True] * (n + 1) # 加一个用不着的0
    flag[0] = flag[1] = False # 方便使用sum计算质数个数
    for i in range(2, math.isqrt(n)+1):
        if flag[i]:
            for j in range(i*i, n+1, i):
                flag[j] = False
    return [i for i, isprime in enumerate(flag) if isprime]
```

## 博弈

```python
@cache
def dfs(s):
    pos_a = []
    start = 0
    while (pos := s.find('++', start)) != -1:
        start = pos + 1
        pos_a.append(pos)

    if len(pos_a) == 0:
        return False # always lose when no next state
    else:
        states = []
        for pos in pos_a:
            new_s = s[:pos] + '--' + s[pos+2:]
            states.append(dfs(new_s))
        if all(states):
            return False # always lose when all next state is alway win
        else:
            return True # always win when at least one next state is alway lose
```

## 几何

```python
def dont_overlap(x1, y1, x2, y2):
    return x2 < y1 or x1 > y2

def overlap(x1, y1, x2, y2):
    return x1 <= y2 and y1 <= x2

def merge(intervals):
    intervals.sort(key=operator.itemgetter(0))
    res = []
    res.append(intervals[0])
    
    for i in range(1, len(intervals)):
        curr = intervals[i]
        last = res[-1]
        if curr[0] <= last[1]:
            # 合并
            last[1] = max(last[1], curr[1])
        else:
            # 新的单独区间
            res.append(curr)
    return res
```

## 排列

```python
def nextPermutation(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    if n == 1: return
    i = n - 2
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    if i == -1:
        start = 0
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    else:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1 
        
        nums[i], nums[j] = nums[j], nums[i]

        start = i + 1
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

def getPermutation(self, n: int, k: int) -> str:
    f = [1]
    for i in range(1, n):
        f.append(f[-1]*i)
    digits = list(range(1, n+1))
    ans = []
    for i in range(n):
        drank = (k - 1) // f[n-1-i]
        k -= drank * f[n-1-i]
        # print(drank, f[n-1-i])
        ans.append(str(digits.pop(drank)))
        # print(ans)
    return "".join(ans)
```
# DP

## 状压DP

```python
#全集 设元素范围从 0 到 3
#括号不可少移位运算优先级低
(1 << 4)-1

#属于
(1101 >> 2) & 1 == 1

#不属于
(1101 >> 2) & 1 == 0

#删除元素
1001|(1 << 2)

#删除元素
1101& ~(1 << 2)

# 删除最小元素
s # 101100
s-1 # 101011 # 最低位的 1 变成 0，同时 1 右边的 0 都取反，变成 1
s&(s-1) # 101000

s.bit_count() # 集合大小（元素个数）

s.bit_length() # 二进制长度（减一得到集合中的最大元素)

(s&-s).bit_length()-1 # 集合中的最小元素

# 只包含最小元素的子集，即二进制最低1及其后面的0，也叫 lowbit，可以用 s & -s 算出。举例说明：
s # 101100
~s # 010011
(~s)+1 # 010100 根据补码的定义，这就是 -s  最低1左侧取反，右侧不变
s & -s # 000100 lowbit

# 设元素范围从 0 到 n −1 挨个判断每个元素是否在集合 s 中：
for i in range(n):
    if (s >> i) & 1:  # i 在 s 中
        # 处理 i 的逻辑

# 设元素范围从 0 到 n −1 从空集枚举到全集
for s in range(1 << n):
    # 处理 s 的逻辑

# 从大到小枚举 s 的所有非空子集
# 简单减一不行10101→10100→10011(不是子集)
# 我们要做的相当于「压缩版」的二进制减法 10101→10100→10001→10000→00101
# 忽略掉 10101中的两个 0，数字的变化和二进制减法是一样的，即111→110→101→100→011
# 如何快速找到下一个子集呢？以10100→10001为例说明
# 普通的二进制减法会把最低位的1变成0，同时1右边的0变成1，即 10100→10011
# 「压缩版」的二进制减法也是类似的，把最低位的1变成0，但同时对于1右边的0只保留在s=10101中的1
# 所以是 10100→10001 怎么保留？&10101就行。
sub = s
while sub:
    # 处理 sub 的逻辑
    sub = (sub - 1) & s

# Gosper's Hack
# 生成n元集合所有 k元子集的算法
s = (1 << k) - 1
while s < (1 << n):
    bits = [i for i, c in enumerate(bin(s)[:1:-1]) if c == '1']
    # bits存储所有不为零的位置
    lb = s & -s
    x = s + lb
    s = (s ^ x) // lb >> 2 | x
```

## 数位DP

```python
def countSpecialNumbers(self, n: int) -> int:
    s = str(n)
    
    @cache  # 记忆化搜索
    def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return 1 if is_num else 0 # 每个合法数字算一个 is_num = False => 全是零 取决于零是否合法
        res = 0
        if not is_num:  # 可以跳过当前数位
            res = f(i + 1, mask, False, False)
        low = 0 if is_num else 1  # 如果前面没有填数字，必须从 1 开始（因为不能有前导零）
        up = int(s[i]) if is_limit else 9  # 如果前面填的数字都和 n 的一样，那么这一位至多填 s[i]（否则就超过 n 啦）
        for d in range(low, up + 1):  # 枚举要填入的数字 d
            if (mask >> d & 1) == 0:  # d 不在 mask 中
                res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
        return res
    
    return f(0, 0, True, False)

def countDigitOne(self, n: int) -> int:
    s = str(n)

    @cache
    def f(i: int, one_num: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return one_num
        res = 0
        if not is_num:
            res = f(i + 1, 0, False, False)
        low = 0 if is_num else 1
        up = int(s[i]) if is_limit else 9
        for d in range(low, up + 1):
            res += f(i + 1, one_num + int(d==1), is_limit and d == up, True)
        return res

    return f(0, 0, True, False)

def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
    s = str(n)

    @cache
    def f(i: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return 1 if is_num else 0
        res = 0
        if not is_num:
            res = f(i + 1, False, False)
        avail = [d for d in digits if int(d) <= int(s[i])] if is_limit else digits
        for d in avail:
            res += f(i + 1, is_limit and int(d) == int(s[i]), True)
        return res

    return f(0, True, False)
```

## 背包

- 学会一维数组方案， 极大节省时间:

```python
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    total = sum(nums)
    if (total + target) < 0 or (total + target) % 2 == 1:
        return 0
    
    p_target = (total + target) // 2
    
    @cache
    def dp(i, target):
        if i < 0:
            return 1 if target == 0 else 0
        else:
            if target < nums[i]:
                return dp(i - 1, target)
            else:
                return dp(i - 1, target - nums[i]) + dp(i - 1, target)

    ans = dp(len(nums) - 1, p_target)
    return ans

def findTargetSumWays(self, nums: List[int], target: int) -> int:
    total = sum(nums)
    if (total + target) < 0 or (total + target) % 2 == 1:
        return 0
    
    p_target = (total + target) // 2
    
    n = len(nums)
    # dp = [[0]*(p_target+1) for i in range(n+1)]
    # dp = [[0]*(p_target+1) for i in range(2)]
    # dp[0][0] = 1
    dp = [0]*(p_target+1)
    dp[0] = 1

    for i, n in enumerate(nums):
        # for c in range(p_target+1):
        #     if c < n:
        #         dp[i+1][c] = dp[i][c]
        #     else:
        #         dp[i+1][c] = dp[i][c] + dp[i][c-n]
        # for c in range(p_target+1):
        #     if c < n:
        #         dp[(i+1)%2][c] = dp[i%2][c]
        #     else:
        #         dp[(i+1)%2][c] = dp[i%2][c] + dp[i%2][c-n]
        # for c in range(p_target, -1, -1):
        #     if c < n:
        #         dp[c] = dp[c]
        #     else:
        #         dp[c] = dp[c] + dp[c-n]
        for c in range(p_target, n-1, -1):
            dp[c] = dp[c] + dp[c-n]
    # print(dp)
    # ans = dp[-1][-1]
    ans = dp[-1]
    return ans

def coinChange(self, coins: List[int], amount: int) -> int:
    n = len(coins)
    
    @cache
    def dp(i, target):
        if i < 0:
            return 0 if target == 0 else 1<<31
        else:
            if target < coins[i]:
                return dp(i - 1, target)
            else:
                return min(dp(i, target - coins[i]) + 1, dp(i - 1, target))
    
    ans = dp(n - 1, amount)
    ans = ans if ans != 1<<31 else -1

    return ans

def coinChange(self, coins: List[int], amount: int) -> int:
    n = len(coins)

    # dp = [[0]* (amount + 1) for i in range(n+1)]
    # dp[0][0] = 0
    dp = [1<<31] * (amount + 1)
    dp[0] = 0

    for i, n in enumerate(coins):
        # 如果是range(1, amount+1), 那么用前i构成的恰好为cost的可行解就被跳过
        # for c in range(amount+1):
        #     if c < n:
        #         dp[i+1][c] = dp[i][c]
        #     else:
        #         dp[i+1][c] = dp[i][c] + dp[i+1][c-n]
        # for c in range(amount+1):
        #     if c < n:
        #         dp[c] = dp[c]
        #     else:
        #         dp[c] = min(dp[c], dp[c-n]+1)
        for c in range(n, amount+1):
            dp[c] = min(dp[c], dp[c-n]+1)
    # ans = dp[-1][-1]
    ans = dp[-1]
    ans = ans if ans != 1<<31 else -1
    return ans

# 多重背包 每种物品有 k_i 个
# 二进制分组优化 把多重背包转化成 0-1 背包模型来求解
# 一个物品可以选7次 =》 可以选 1个物品 2个物品 4个物品 至多一次
```

- 求max/min的模型里：
    - 求体积**恰好**为j：
        - 求max, dp【0】 = 【0】+【-inf】\* t
        - 求min, dp【0】 = 【0】+【inf】\* t
        - 最终f【j】代表体积恰好为j时的价值极值。
    - 求体积**至多**为j时:
        - dp【0】 = 【0】+【0】\* t
        - 最终f【j】代表体积至多为j时的价值最大值
    - 求体积**至少**为j时:
        - dp【0】 = 【0】+【inf】\* t
        - 同时遍历体积需要修改循环下界v->0、转移需要修改为从max(0,j-v)
        - 最终f【j】代表体积至少为j时的价值最小值
        - f【0】始终为0

```python
#01背包改为
@cache
def dp(i, target):
    if i < 0:
        return 0 if target <= 0 else 1<<31
    else:
        return min(dp(i - 1, target - coins[i]) + 1, dp(i - 1, target))

dp = [1<<31] * (amount + 1)
dp[0] = 0
for i, n in enumerate(coins):
    for c in range(amount, -1, -1):
        dp[c] = min(dp[c], dp[max(c-n, 0)] + 1)
#完全背包改为
@cache
def dp(i, target):
    if i < 0:
        return 0 if target <= 0 else 1<<31
    else:
        return min(dp(i, target - coins[i]) + 1, dp(i - 1, target))

dp = [1<<31] * (amount + 1)
dp[0] = 0
for i, n in enumerate(coins):
    for c in range(amount+1):
        dp[c] = min(dp[c], dp[max(c-n, 0)] + 1)
```
- 求方案数的模型里（一般要取模）:
    - 求体积**恰好**为j：
       - 求max, f【0】 = 【1】+【0】\* t
       - 最终f【j】代表体积恰好为j时的方案数。
    - 求体积**至多**为j时:
       - f【0】 = 【1】+【1】\* t  
       - 最终f【j】代表体积至多为j时的方案数。
    - 求体积**至少**为j时:
       - f【0】 = 【1】+【0】\* t 
       - 同时遍历体积需要修改循环下界v->0、转移需要修改为从max(0,j-v)
        - 最终f【j】代表体积至少为j时的方案数
        - f【0】始终为0

```python
# 01背包改为
@cache
def dp(i, target):
    if i < 0:
        return 1 if target <= 0 else 0
    else:
        return dp(i - 1, target - coins[i]) + 1 + dp(i - 1, target)

dp = [0] * (amount + 1)
dp[0] = 1
for i, n in enumerate(coins):
    for c in range(amount, -1, -1):
        dp[c] = dp[c] + dp[max(c-n, 0)] + 1
#完全背包改为
@cache
def dp(i, target):
    if i < 0:
        return 1 if target <= 0 else 0
    else:
        return dp(i, target - coins[i]) + 1 + dp(i - 1, target)

dp = [0] * (amount + 1)
dp[0] = 1
for i, n in enumerate(coins):
    for c in range(amount+1):
        dp[c] = dp[c] + dp[max(c-n, 0)] + 1
ans = dp[-1]
```

# 字符串

```python
str.split(sep=',', maxsplit=1)
# maxsplit control split times

str.zfill(width)
# "-42".zfill(5) => '-0042'

str.partition(sep)
# "abc".partition("b") => "a", "b", "c"
# "abc".partition("d") => "abc", "", ""

str.startswith(suffix[, start[, end]])
str.endswith(suffix[, start[, end]])
# can provide multiple suffix

s[start:end]
# slicing in Python is O(n)

str.find(sub[, start[, end]])
# return -1 if not find

s_suffix = [s[i:] for i in range(len(x))]
s_prefix = [s[:i+1] for i in range(len(s))]

def is_palindrome(s):
    return s == reversed(s)

def build_prefix_table(pattern):
    prefix_table = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            prefix_table[i] = length
            i += 1
        else:
            if length != 0:
                length = prefix_table[length - 1]
            else:
                prefix_table[i] = 0
                i += 1

    return prefix_table


def kmp_search(text, pattern):
    m = len(pattern)
    n = len(text)
    prefix_table = build_prefix_table(pattern)
    i = 0  # index for text
    j = 0  # index for pattern
    indices = []  # stores the indices of pattern occurrences

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

            if j == m:
                indices.append(i - j)
                j = prefix_table[j - 1]
        else:
            if j != 0:
                j = prefix_table[j - 1]
            else:
                i += 1

    return indices

```

# 前缀树

```python
class TrieNode:
    def __init__(self, char):
        self.val = char
        self.edges = {}
        self.is_word_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode(None)
    
    def insert(self, word):
        '''Inserts word into the trie'''
        curr = self.root
        for char in word:
            if char not in curr.edges:
                curr.edges[char] = TrieNode(char)
            curr = curr.edges[char]
        curr.is_word_end = True
    
    def search(self, word):
        '''Returns true if word is in the trie'''
        curr = self.root
        for char in word:
            if char not in curr.edges:
                return False
            curr = curr.edges[char]
        return curr.is_word_end

    def prefix(self, word):
        curr = self.root
        ans = []
        path = deque()
        for char in word:
            if char not in curr.edges:
                return ans
            curr = curr.edges[char]
            path.append(char)
        
        def dfs(node):
            nonlocal path, ans
            if node.is_word_end:
                ans.append(''.join(path))
            for c in node.edges:
                path.append(c)
                dfs(node.edges[c])
                path.pop()
        
        dfs(curr)
                
        return ans
```

# 单调栈 单调队列

```python
# 核心思想: 利用单调性避免栈、队列大小达到O(N) 进而导致O(N^2)做法存在大量无效比较
# find the previous less element of each element in a vector with O(n) time
s = deque()
previous_less = [-1] * len(arr)
for i, x in enumerate(arr):
    while len(s) > 0 and arr[s[-1]] > x:
        s.pop()
    previous_less[i] = -1 if len(s) == 0 else s[-1]
    s.append(i)
# find the next less element of each element in a vector with O(n) time:
s = deque()
next_less = [-1] * len(arr)
for i, x in enumerate(arr):
    while len(s) > 0 and arr[s[-1]] > x:
    	next_less[s.pop()] = i
    s.append(i)
# find size K sliding window max
dq = deque()
res = []
for i in range(len(nums)):
    if dq and dq[0] == i - k:
        dq.popleft()
    while dq and nums[i] >= nums[dq[-1]]:
        dq.pop()
    dq.append(i)
    if i >= k - 1:
        res.append(nums[dq[0]])
```

# 并查集

```python
class DisjointSet:
    def __init__(self):
        self.count = 0
        self.parent = dict()
        self.rank = dict()

    def add(self, p):
        self.parent[p] = p
        self.rank[p] = 1

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1
        self.count -= 1
```

# 图

```python
def topologicalSort():
    indegree = [0] * numCourses
    adj = defaultdict(list)
    
    for prerequisite in prerequisites:
        adj[prerequisite[1]].append(prerequisite[0])
        indegree[prerequisite[0]] += 1

    queue = deque()
    for i in range(numCourses):
        if indegree[i] == 0:
            queue.append(i)
	
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
'''
e:边集
adj:邻接表
s:起点
dis:最短路长度
'''
dis = [[(1<<31)] * n for k in range(n)]
for k in range(n):
    dis[k][k] = 0
for u, v, w in edges:
    dis[u][v] = w
    dis[v][u] = w
        
#floyd 多源最短路算法 可以处理负权边不能处理负环
# 空间复杂度O(N^2)，时间复杂度O(N^3)
for k in range(n):
    for x in range(n):
        for y in range(n):
            dis[x][y] = min(dis[x][y], dis[x][k] + dis[k][y])

adj = defaultdict(list)
for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))


# Bellman-Ford 有负权的图的最短路
# 时间复杂度为O(NE)，e为图的边数，在图为稠密图的时候，是不可接受的。复杂度太高。
dis = [(1<<31)] * n
dis[s] = 0
for i in range(n):
    flag = False
    for u in adj:
        for v, w in adj[u]:
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                flag = True
    # 没有可以松弛的边时就停止算法
    if flag == False:
        break
    # 第 n 轮循环仍然可以松弛时说明 s 点可以抵达一个负环
return flag

# Bellman-Ford + 队列优化 => spfa
dis = [(1<<31)] * n
dis[s] = 0
queue = deque([s])
cnt = [0] * n
vis = [0] * n
vis[s] = 1
while queue:
    u = queue.popleft()
    vis[u] = 0
    for v, w in adj[u]:
        if dis[u] + w < dis[v]:
            dis[v] = dis[u] + w
            cnt[v] = cnt[u] + 1 # 记录最短路经过的边数
            if cnt[v] >= n:
                return False
            # 在不经过负环的情况下，最短路至多经过 n - 1 条边
            # 因此如果经过了多于 n 条边，一定说明经过了负环
            if not vis[v]:
                queue.append(v)
                vis[v] = True

dis = [(1<<31)] * n
dis[s] = 0

# dijkstra 单源最短路算法，其要求图中的边全部非负
# 使用二叉堆优化后的Dijkstra算法的复杂度为O((E+N)logN)，因此该优化适合于稀疏图
# 如果是稠密图极端情况E = n*(n-1)/2，这时候时间复杂度就退化为O(N^2logN)了, 得不偿失。
q = [(0, s)]
vis = set()
while q:
    _, u = heapq.heappop(q)
    if u in vis:
        continue
    vis.add(u)
    for v, w in adj[u]:
        if dis[v] > dis[u] + w:
            dis[v] = dis[u] + w
            heapq.heappush(q, (dis[v], v))
```

# 堆
- 建堆是O(N)所以仍然比排序在Top K问题上有优势

```python
# Top K/Kth largest
# construct maximum heap and pop first K element from it
heapq.heapify([-n for n in nums])
for i in range(k):
    # top i+1 th largest element
    e = -heapq.heappop(heap, n)
# Time complexity: O(KlogN+N)
# Space complexity: O(N)

# filter through array and maintain K largest element dynamically
heap = []
for n in nums:
    if len(heap) < k:
        heapq.heappush(heap, n)
    else:
        heapq.heappushpop(heap, n)
# top kth largest element
heap[0]
# Time complexity: O(NlogK)
# Space complexity: O(K)

# Top K/Kth smallest
heapq.heapify([n for n in nums])
for i in range(k):
    e = heapq.heappop(heap, n)

for n in nums:
    if len(heap) < k:
        heapq.heappush(heap, -n)
    else:
        heapq.heappushpop(heap, -n)
-heap[0]
```


# 二叉搜索树

- C++ OrderedMap / Java TreeMap 在python中最接近的替代品 `from sortedcontainers import SortedList, SortedDict, SortedSet`
- 内部并不是用二叉搜索树、平衡树，但是从概念上和复杂度上和二叉树更接近 大部分复杂度O(logn)
- `SortedList`用起来和List差不多
    - 不要用`key`参数很容易弄出不满足全序的怪胎，自定义class实现eq和lt
    - 不能利用下标赋值，可以用下标取值
    - 使用索引取值,  使用`in`搜索, 使用`index`搜索的时间复杂度是O(logn) 使用`bisect_left`，`bisect_right`搜索的时间复杂度是O(logn)
    - 按值删 discard(value)跟remove(value)的差别在前者在移除不存在的元素时不会引发错误 按索引删pop(index)
    - 没有append()/extend()/insert()，因为加入新的元素都有自己应该在的位置，应该使用add() 一次加入多个可以用update(),或着使用`sl += some_iterable`
- `SortedDict`基于`SortedList`实现只是额外存储对应的value，但只对key排序
    - 可以用index取值, 但要用peekitem(), O(logn)
    - 插入/刪除是O(logn)
    - 只根据key排序
- 非递归遍历

```python
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    ans = []
    stack = deque([root])
    while stack:
        cur = stack.pop()
        if cur is not None:
            ans.append(cur.val)
            stack.append(cur.right)
            stack.append(cur.left)
    return ans

def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    cur = root
    stack = deque()
    res = []
    while len(stack) > 0 or cur is not None:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        res.append(cur.val)
        cur = cur.right
    return res

def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    ans = []
    stack = deque([root])
    while stack:
        cur = stack.pop()
        if cur is not None:
            ans.append(cur.val)
            stack.append(cur.left)
            stack.append(cur.right)
    return ans[::-1]
```

- 增删查改

```python
def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
    if not root:
        return TreeNode(val)
    
    if val > root.val:
        # insert into the right subtree
        root.right = self.insertIntoBST(root.right, val)
    else:
        # insert into the left subtree
        root.left = self.insertIntoBST(root.left, val)
    return root

def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if root is None:
        return None
    
    if root.val < val:
        return self.searchBST(root.right, val)
    elif root.val == val:
        return root
    else:
        return self.searchBST(root.left, val)

def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    if root is None:
        return None
    
    if root.val == key:
        if root.left is not None:
            p = root.left
            while p.right is not None:
                p = p.right
            root.val = p.val
            root.left = self.deleteNode(root.left, root.val)
        elif root.right is not None:    
            p = root.right
            while p.left is not None:
                p = p.left
            root.val = p.val
            root.right = self.deleteNode(root.right, root.val)
        else:
            return None
    elif root.val < key:
        root.right = self.deleteNode(root.right, key)
    else:
        root.left = self.deleteNode(root.left, key)
    return root

```
