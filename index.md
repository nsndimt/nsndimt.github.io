---
title: LeetCode Note for nsndimt
layout: default
---

# Python语言
- `dims=[state_dim, *dims, action_dim]`构造数组而不是`dims=[state_dim] + dims + [action_dim]`
- 二维数组创建和下标访问就用`[[0]*1000 for i in range(1000)]`可以了除非有向量化运算才用`numpy`
- `a, b = a + b, a - b` 的计算顺序为：
	- a + b
	- a - b
	- 赋值 a
	- 赋值 b
- `a = b = 1 + 1` 的计算顺序为：
	- 1 + 1
	- 赋值 a
	- 赋值 b
 - if `i < len(arr) and arr[i] > 0` 的计算顺序为：
     -  i < len(arr)
     -  arr[i] > 0 如果第一个条件为假则不计算(short circuiting)
- `a <= b <= c` 的计算顺序为：
	- a <= b
	- b <= c
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
        - [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
        - [list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D
- You need to use nonlocal whenever you want to assign to a nonlocal variable in a local scope, exactly analogous to global
# 排序
- 大部分有序排序
	- 1、最理想情况（数据预先已排好序），插入和冒泡都只需要进行n次循环和比较就结束了，不需要进行数据交换（传值），而选择要进行(n^2)/2次循环和比较，显然选择明显落后于冒泡和插入
	- 2、平均情况（数据完全随机），插入要进行(n^2)/4次循环和比较，以及同是(n^2)/4次的数据复制（传值）而冒泡要进行(n^2)/2次循环和比较，(n^2)/4次交换，每次交换等于3次数据复制（传值），因此它的循环比较次数和和数据复制次数分别是插入的2倍和3倍，因此冒泡的耗时是插入的2-3倍之间
 	- 如果排序对象是简单数据，每次比较操作的耗时大于数据复制，则冒泡的耗时会比较接近插入的2倍；如果排序对象是很长的数据结构，每次复制数据的耗时远大于比较，则冒泡的耗时会更接近插入的3倍。
- 前K大 1. 冒泡 K 遍 2. 维护大小为K最大堆 3.快排
- 第K大 1. 快排
- 逆序对: 归并排序
```Python
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
        arr[idx], arr[min_idx]) = arr[min_idx], array[idx]

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
```Python
def cmp(a, b):
    if a[0] > b[0]:
        return 1
    elif a[0] < b[0]:
        return -1
    else:
        return 0

sorted([(1, 2), (4, 2)], key=functools.cmp_to_key(cmp)))
```
- 自定义`heapq`和`SortedContainer`比较函数
```Python
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
```Python 
prefixsum = [0] + list(accumulate(arr))
def query(i, j):
    # 查询的是双闭区间[i, j]的区间和
    return prefixsum[j+1] - prefixsum[i]

suffixsum = list(accumulate(arr)) + [0]
def query(i, j):
    # 查询的是双闭区间[i, j]的区间和
    return suffixsum[i] - suffixsum[j+1]

diff = [arr[0]] + [arr[i] - arr[i - 1] for i in range(1, len(arr))]
def modify(i, j, value):
    # 取[i~j]的双闭区间进行区间修改
    diff[i] += value  # 复原时, arr[i]之后的数都会 + value
    if j + 1 < len(self.diff):
        self.diff[j + 1] -= value
# 一连串的modify最后recover
def recover():
    # 复原修改后的数组
    res = [self.diff[0]]
    for i in range(1, len(self.diff)):
        res.append(res[-1] + self.diff[i])
    return res
```
# 树状数组
![image](https://github.com/nsndimt/leetcode/assets/8330249/9ec0c66e-7341-4e82-8231-7fa2597dec90)
```Python
def lowbit(x):
    """
    x 的二进制中，最低位的 1 以及后面所有 0 组成的数。
    lowbit(0b01011000) == 0b00001000
    lowbit(0b01110010) == 0b00000010
    """
    return x & -x

class NumArray:

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
```Python
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
        while left <= right and s - nums[left] >= target:
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
                # have finded every valid largest window ended with right
                ans += right - left + 1
                break
            p //= nums[left]
            left += 1
    return and

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
```Python
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
  
| position            | bisect                          | minium possible value | maxium possible value |
| -----------         | -----------                     | -----------           | -----------           |
| last less or equal  | bisect.bisect_right(arr, v) - 1 | -1 (not find)         | len(arr) - 1          |
| first great         | bisect.bisect_right(arr, v)     | -1                    | len(arr) (not find)   |
| last less           | bisect.bisect_left(arr, v) - 1  | -1 (not find)         | len(arr) - 1          |
| first great or equal| bisect.bisect_right(arr, v)     | -1                    | len(arr) (not find)   |

| position            | if condition    | return                       | red         | blue        |
| -----------         | -----------     | -----------                  | ----------- | ----------- |
| last less or equal  | arr[mid] <= v   | left                         | <= v        | > v         |
| first great         | arr[mid] <= v   | right                        | <= v        | > v         |
| last less           | arr[mid] < v    | left                         | < v         | >= v        |
| first great or equal| arr[mid] < v    | right                        | < v         | >=v         |
| equal               | arr[mid] <> v   | -1 all colored => not found  | < v         | >=v         |

## 开区间
- 因为未检查区间 $(left, right)$ 为开区间 所以`left = -1; right = len(arr)`
- 红区间: $[0, left]$ 蓝区间: $[right, len(arr) - 1]$
- 因为未检查区间 $(left, right)$ 为开区间 所以`left + 1 == right`时区间为空 `while left + 1 < right:`
- `left = -1; right = 1` => `mid = (left + right) // 2`
- 因为未检查区间 $(left, right)$ 为开区间 所以 `left = mid`和`right = mid`不会导致mid留在未检查区间里

## 左闭右开区间
- 因为未检查区间 $[left, right)$ 为左闭右开区间 所以`left = 0; right = len(arr)`
- 红区间: $[0, left)$ 蓝区间: $[right, len(arr) - 1]$
- 因为未检查区间 $[left, right)$ 为左闭右开区间 所以`left == right`时区间为空 `while left < right:`
- `left = 0; right = 1` => `mid = (left + right) // 2`
- 因为未检查区间 $[left, right)$ 为左闭右开区间 所以 `left = mid + 1`和`right = mid`不会导致mid留在未检查区间里

## 闭区间
- 因为未检查区间 $[left, right]$ 为闭区间 所以`left = 0; right = len(arr) - 1`
- 红区间: $[0, left)$ 蓝区间: $(right, len(arr) - 1]$
- 因为未检查区间 $[left, right]$ 为闭区间 所以`left == right + 1`时区间为空 `while left <= right:`
- `left = 0; right = 0` => `mid = (left + right) // 2`
- 因为未检查区间 $[left, right]$ 为闭区间 所以 `left = mid + 1`和`right = mid - 1`不会导致mid留在未检查区间里

## 左开右闭区间
- 因为未检查区间 $(left, right]$ 为左开右闭区间 所以`left = -1; right = len(arr) - 1`
- 红区间: $[0, left]$ 蓝区间: $(right, len(arr) - 1]$
- 因为未检查区间 $(left, right]$ 为左开右闭区间 所以`left == right`时区间为空 `while left < right:`
- `left = -1; right = 0` => `mid = (left + right + 1) // 2`
- 因为未检查区间 $(left, right]$ 为左开右闭区间 所以 `left = mid`和`right = mid - 1`不会导致mid留在未检查区间里

# 递归
## 子集型
```Python
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
## 组合型
```Python
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
## 排列
```Python
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
```Python
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
```Python
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
```Python
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
# DP
## 状压DP
参考[https://leetcode.cn/circle/discuss/CaOJ45/]
## 数位DP
```Python
def countSpecialNumbers(self, n: int) -> int:
    s = str(n)
    
    @cache  # 记忆化搜索
    def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return int(is_num)  # is_num 为 True 表示得到了一个合法数字
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
```
## 背包
- 求max/min的模型里：
	- 求体积`恰好`为j：
	- 求max, f = 【0】+【-inf】*t
	- 求min, f = 【0】+【inf】*t
	- 最终f【j】代表体积恰好为j时的价值极值。
	---
	- 求体积`至多`为j时:
	- f【0】 = 【0】+【0】*t  (max/min)
	- 最终f【j】代表体积`至多`为j时的价值极值
	---
	- 求体积`至少`为j时:
	- f【0】 = 【0】+【0】*t  (max/min)
	- 同时遍历体积需要修改循环下界v->0、转移需要修改为从max(0,j-v),即
		`for j in range(self.vol, -1, -1):f【j】 = merge(f【j】, f【max(j - v,0)】 + w)  # 01背包`  
		`for j in range(self.vol+1):f【j】 = merge(f【j】, f【max(j - v,0)】 + w)  # 完全背包`  
	- 最终f【j】代表体积`至少`为j时的价值极值
--- 
- 求方案数的模型里（一般要取模）:
	- 求体积`恰好`为j：
	- 求max, f = 【1】+【0】*t
	- 最终f【j】代表体积恰好为j时的方案数。
	---
	- 求体积`至多`为j时:
	- f = 【1】+【1】*t  
	- 最终f【j】代表体积`至多`为j时的方案数。
	---
	- 求体积`至少`为j时:
	- f = 【1】+【0】*t 
	- 同时遍历体积需要修改循环下界v->0、转移需要修改为从max(0,j-v),即
		`for j in range(self.vol, -1, -1):f【j】 += f【max(j - v,0)】  # 01背包`  
		`for j in range(self.vol+1):f【j】 += f【max(j - v,0)】  # 完全背包`  
	- 最终f【j】代表体积`至多少`为j时的方案数
```Python
@cache
def complete_backpack(i, target):
    if i < 0:
        return 0 if target == 0 else 1000000
    else:
        if target < coins[i]:
            return complete_backpack(i - 1, target)
        else:
            return min(complete_backpack(i, target - coins[i]) + 1, complete_backpack(i - 1, target))

dp = [[0]+[1000000] * amount for i in range(n+1)]
for i, n in enumerate(coins):
    for c in range(amount+1):
        if c < n:
            dp[i+1][c] = dp[i][c]
        else:
            dp[i+1][c] = min(dp[i][c], dp[i+1][c-n]+1)

@cache
def zero_one_backpack(i, target):
    if i < 0:
        return 1 if target == 0 else 0
    else:
        if target < nums[i]:
            return zero_one_backpack(i - 1, target)
        else:
            return zero_one_backpack(i - 1, target - nums[i]) + zero_one_backpack(i - 1, target)

dp = [[1]+[0]*p_target for i in range(n+1)]
for i, n in enumerate(nums):
    # 如果是range(1, target+1), 那么用前i构成的恰好为cost的可行解就被跳过
    for c in range(target+1):
        if c < n:
            dp[i+1][c] = dp[i][c]
        else:
            dp[i+1][c] = dp[i][c] + dp[i][c-n]
```
# 字符串
```Python
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
```Python
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
```
# 单调栈 单调队列
```Python
# 核心思想: 利用单调性避免栈、队列大小达到O(N) 进而导致O(N^2)做法存在大量无效比较
# find the previous less element of each element in a vector with O(n) time
s = deque()
previous_less = [-1] * len(arr)
for i, x in enumerate(arr):
    while len(s) > 0 and arr[s[-1]] > x:
        s.pop()
    previous_less[i] = -1 if len(s) == 0 else s[-1]
    s.push(i)
# find the next less element of each element in a vector with O(n) time:
s = deque()
next_less = [-1] * len(arr)
for i, x in enumerate(arr):
    while len(s) > 0 and arr[s[-1]] > x:
    	next_less[s.pop()] = i
    s.push(i)
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
```Python
class DisjointSet:
    def __init__(self, n):
        # number of components
        self.count = n 
        self.parent = list(range(n))
        self.rank = [0] * n  

    def find(self, p):
        if self.parent[p] != value:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if   self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1
        self.count -= 1
```
# 图
```Python
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

def floyd():
    for k in range(1, n + 1):
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                f[x][y] = min(f[x][y], f[x][k] + f[k][y])
'''
e:边集
adj:邻接表
s:起点
dis:最短路长度
'''
dis = [-(1<<31)] * n
for u, v, w in e:
    dis[u][v] = w
def floyd():
    for k in range(1, n + 1):
        for x in range(1, n + 1):
            for y in range(1, n + 1):
                dis[x][y] = min(dis[x][y], dis[x][k] + dis[k][y])

#有负权的图的最短路
dis = defaultdict(lambda: -(1<<31))
dis[s] = 0
def bellmanford(n, s):
    dis[s] = 0
    for i in range(1, n + 1):
        for u in adj:
            for v, w in adj[u]:
                if dis[v] > dis[u] + w:
                    dist[v] = dist[u] + w

dis = defaultdict(lambda: -(1<<31))
dis[s] = 0
def spfa(s):
    queue = deque([s])
    vis = set([s])
    while queue:
        u = queue.popleft()
        vis.remove(u)
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if v not in vis:
                    queue.append(v)
                    vis.add(u)
# 非负权图上单源最短路径
dis = defaultdict(lambda: -(1<<31))
dis[s] = 0
def dijkstra(s):
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
# BST
- C++ OrderedMap / Java TreeMap 在python中最接近的替代品 `from sortedcontainers import SortedList, SortedDict, SortedSet`
- 内部并不是用二叉搜索树、平衡树，但是从概念上和复杂度上和二叉树更接近 大部分复杂度log(n)
- `SortedList`用起来和List差不多
    - 不要用`key`参数很容易弄出不满足全序的怪胎，自定义class实现eq和lt
    - 不能利用下标赋值，可以用下标取值
    - 使用索引取值,  使用`in`搜索, 使用`index`搜索的时间复杂度是O(lg(n)) 使用`bisect_left`，`bisect_right`搜索的时间复杂度是O(lg(n))
    - discard()跟remove()的差别在前面在移除不存在的元素时不会引发错误
    - 没有append()/extend()/insert()，因为加入新的元素都有自己应该在的位置，应该使用add()。一次加入多个可以用update(),或着使用`sl += some_iterable`
- `SortedDict`基于`SortedList`实现只是额外存储对应的value，但只对key排序
    - 可以用index取值, 但要用peekitem(), O(ln(n))
    - 插入/刪除是O(ln(n))
    - 对key排序
