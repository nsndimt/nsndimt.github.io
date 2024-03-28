import math
import string
import bisect
import re
import operator
import heapq
import queue

from itertools import combinations, permutations
from functools import cache
from collections import defaultdict
from collections import OrderedDict
from collections import deque
from collections import Counter

import pdb

def nextPermutation(nums):
    n = len(nums)
    if n == 1:
        return None
    i = n - 2
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    if i == -1:
        return None
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
    
    
        
print('a' < 'b')