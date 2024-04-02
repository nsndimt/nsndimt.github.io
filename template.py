from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
import time

from typing import *


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def stringToIntegerList(input):
    return json.loads(input)


def stringToListNode(input):
    # Generate list from the input
    numbers = stringToIntegerList(input)

    # Now convert that list into linked list
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next
    return ptr


def prettyPrintLinkedList(node):
    while node and node.next:
        print(str(node.val) + "->", end="")
        node = node.next

    if node:
        print(node.val)
    else:
        print("Empty LinkedList")


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def treeNodeToString(root):
    if not root:
        return "[]"
    output = ""
    queue = [root]
    current = 0
    while current != len(queue):
        node = queue[current]
        current = current + 1

        if not node:
            output += "null, "
            continue

        output += str(node.val) + ", "
        queue.append(node.left)
        queue.append(node.right)
    return "[" + output[:-2] + "]"


def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(",")]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


def prettyPrintTree(node, prefix="", isLeft=True):
    if not node:
        print("Empty Tree")
        return

    if node.right:
        prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)

    print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))

    if node.left:
        prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)


if __name__ == "__main__":
    testcase = """
    2245047
    2908305
    10
    """

    def readlines():
        for line in testcase.splitlines():
            line = line.strip()
            if line:
                yield line

    class Solution:
        def minimumOperations(self, num: str) -> int:
            ans = len(num)
            num_int = int(num)

            queue = deque()
            vis = set()
            queue.append((num_int, 0))
            vis.add(num_int)

            while queue:
                x, d = queue.popleft()
                if x % 25 == 0:
                    ans = min(ans, d)
                    break

                if x % 10 in [0, 5] and x >= 10:
                    new_x = (x // 100) * 10 + x % 10
                    if new_x not in vis:
                        queue.append((new_x, d + 1))
                        vis.add(new_x)
                new_x = x // 10
                if new_x not in vis:
                    queue.append((new_x, d + 1))
                    vis.add(new_x)

            return ans

    lines = readlines()
    while True:
        try:
            args = next(lines)
            start_time = time.time()
            ret = Solution().minimumOperations(args)
            end_time = time.time()
            print(f"take {1000*(end_time - start_time):.4f} ms")
            print("ret", ret)
        except StopIteration:
            break
