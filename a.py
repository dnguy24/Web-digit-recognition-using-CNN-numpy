# # # import numpy as np
# # # import math
# # # import tensorflow as tf
# # # import sys
# # # def normal(shape, scale=0.05):
# # #     '''
# # #     :param shape:
# # #     :param scale:
# # #     :return:
# # #     '''
# # #     return np.random.normal(0, scale, size=shape)
# # # def uniform(shape, scale=0.05):
# # #     '''
# # #     :param shape:
# # #     :param scale:
# # #     :return:
# # #     '''
# # #     return np.random.uniform(-scale, scale, size=shape)
# # # def get_fans(shape):
# # #     '''
# # #     :param shape:
# # #     :return:
# # #     '''
# # #     fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
# # #     fan_out = shape[1] if len(shape) == 2 else shape[0]
# # #     return fan_in, fan_out
# # # def he_normal(shape):
# # #     '''
# # #     A function for smart normal distribution based initialization of parameters
# # #     [He et al. https://arxiv.org/abs/1502.01852]
# # #     :param fan_in: The number of units in previous layer.
# # #     :param fan_out: The number of units in current layer.
# # #     :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in]
# # #     '''
# # #     fan_in, fan_out = get_fans(shape)
# # #     scale = np.sqrt(2. / fan_in)
# # #     shape = (fan_out, fan_in) if len(shape) == 2 else shape         # For a fully connected network
# # #     bias_shape = (fan_out, 1) if len(shape) == 2 else (
# # #         1, 1, 1, shape[3])   # This supports only CNNs and fully connected networks
# # #     return normal(shape, scale), uniform(bias_shape)
# # #
# # # a = he_normal(np.array([2,3]))
# # # def calibrated_initializer(shape, scale=0.05):
# # #     n, _ = get_fans(shape)
# # #     return np.random.normal(0, 0.05, size=shape) * np.sqrt(2.0/n)
# # # b = calibrated_initializer(np.array([2,3]))
# # #
# # # # a = np.array([2, 3, 4, 5])
# # # # print(np.where(a>3, a, np.exp(a)))
# # # # print(np.exp(a))
# # # #
# # # # (_, train_labels), (_, test_labels) = tf.keras.datasets.mnist.load_data()
# # # # train_labels = train_labels[:20]
# # # # print(train_labels)
# # # # train_labels = tf.one_hot(train_labels, depth=10)
# # # # print(np.array(train_labels).T)
# # #
# # # # a = np.array([[2,3,4],[5,6,7]])
# # # # print(np.where(a>2, a, 0))
# # # # def maxSubArray(nums) -> int:
# # # #     maxendinghere = nums[0]
# # # #     maxsofar = nums[0]
# # # #     for i in range(1, len(nums)):
# # # #         print(nums[i])
# # # #         maxendinghere = max(maxendinghere + nums[i], nums[i])
# # # #         print("temp:",maxendinghere)
# # # #         maxsofar = max(maxendinghere, maxsofar)
# # # #         print("max:",maxsofar)
# # # #     return maxsofar
# # # # def mypow(x, n):
# # # #     if n == 0:
# # # #         return 1
# # # #     m = x
# # # #     for i in range(abs(n)-1):
# # # #         x = x*m
# # # #     if n<0:
# # # #         x=1/x
# # # #     return x
# # # # print(mypow(2,-2))
# #
# # def bubble(nums):
# #     for i in range(len(nums)):
# #         for j in range(len(nums)-1):
# #             if nums[j]>nums[j+1]:
# #                 nums[j], nums[j+1] = nums[j+1], nums[j]
# # def selection(nums):
# #     for i in range(len(nums)):
# #         minindex = i
# #         for j in range(i+1, len(nums)):
# #             if nums[j]<nums[minindex]:
# #                 minindex = j
# #         nums[minindex], nums[i] = nums[i], nums[minindex]
# # def insertion(nums):
# #     for i in range(1, len(nums)):
# #         key = nums[i]
# #         j=i-1
# #         while(j>=0 and nums[j]>key):
# #             print(nums[j])
# #             nums[j+1] = nums[j]
# #             j-=1
# #             print(j)
# #         nums[j+1] = key
# # def merge(nums1, m, nums2, n):
# #     while m > 0 and n > 0:
# #         if nums1[m - 1] >= nums2[n - 1]:
# #             nums1[m + n - 1] = nums1[m - 1]
# #             m -= 1
# #         else:
# #             nums1[m + n - 1] = nums2[n - 1]
# #             n -= 1
# #     print(nums1)
# #     if n > 0:
# #         nums1[:n] = nums2[:n]
# # def generate(numRows):
# #     res = [[1]]
# #     for i in range(1, numRows):
# #         a = res[-1] + [0]
# #         b = [0] + res[-1]
# #         res += list(map(lambda x, y: x + y, a,b))
# #         print(res)
# #     return res[:numRows]
# # a = [1,2,3,4]
# # # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, x):
# #         self.val = x
# #         self.next = None
# # a = ListNode(1)
# # b = ListNode(1)
# # a.next = b
# # c = ListNode(2)
# # b.next = c
# # d = ListNode(2)
# # c.next = d
# #
# # def printNode(head):
# #     curr = head
# #     while(curr):
# #         print(curr.val)
# #         curr = curr.next
# # def deleteDuplicates(head):
# #     curr = head
# #     temp = ListNode(0)
# #     temp.next = head
# #     dummyNode = temp.next
# #     while (curr):
# #         if (curr.val != dummyNode.val):
# #             dummyNode.next = curr
# #             dummyNode = dummyNode.next
# #
# #         curr = curr.next
# #     return temp.next
# #
# # def rob(nums) -> int:
# #     op = 0
# #     cl = 0
# #     temp = 0
# #     for i in range(len(nums)):
# #         temp = cl
# #         if (op + nums[i] > cl):
# #             cl = op + nums[i]
# #             print(cl)
# #         op = temp
# #
# #     return cl
# # def fibo(n):
# #     if n == 0:
# #         return 0
# #     if n == 1:
# #         return 1
# #     if n == 2:
# #         return 2
# #     arr = [0, 1, 2]
# #     for i in range(3, n+1):
# #         c = arr[i - 1] + arr[i - 2]
# #         print(c)
# #         arr.append(c)
# #     return arr[-1]
# # def computelps(pattern):
# #     m = len(pattern)
# #     i = 1
# #     j = 0
# #     lps = [0]+[None]*(m-1)
# #     print(lps)
# #     while(i<m):
# #         if pattern[i]==pattern[j]:
# #             lps[i] = j+1
# #             j+=1
# #             i+=1
# #         else:
# #             if(j!=0):
# #                 j = lps[j-1]
# #             else:
# #                 lps[i] = 0
# #                 i+=1
# #     return lps
# # a = [1,2,3,4,5]
# # b = 2
# # def plusone(b):
# #     b = b+1
# # def plus(a):
# #     for i in range(len(a)):
# #         a[i] = a[i]+1
# #
# #
# # def computelps(pattern):
# #     m = len(pattern)
# #     i = 1
# #     j = 0
# #     lps = [0]+[None]*(m-1)
# #     while(i<m):
# #         if pattern[i]==pattern[j]:
# #             lps[i] = j+1
# #             i+=1
# #             j+=1
# #         else:
# #             if(j!=0):
# #                 j = lps[j-1]
# #             else:
# #                 lps[i] = 0
# #                 i+=1
# #     return lps
# #
# #
# # def KMP(string, pattern):
# #     lps = computelps(pattern)
# #     print(lps)
# #     i = 0
# #     j = 0
# #     m = len(pattern)
# #     while(i<len(string)):
# #         if string[i] == pattern[j]:
# #             i+=1
# #             j+=1
# #             if j==m:
# #                 return i-j
# #         elif string[i]!=pattern[j]:
# #             if(j!=0):
# #                 j = lps[j-1]
# #             else:
# #                 i+=1
# #     return -1
# #
# #
# # def longestCommonPrefix(strs):
# #     """
# #     :type strs: List[str]
# #     :rtype: str
# #     """
# #     if not strs:
# #         return ""
# #     shortest = min(strs, key=len)
# #     for i, ch in enumerate(shortest):
# #         for other in strs:
# #             if other[i] != ch:
# #                 return shortest[:i]
# #     return shortest
# # def rotatedDigits(S, N: int) -> int:
# #     count = 0
# #     for i in range(S, N+1):
# #         if rotate(i):
# #             count+=1
# #     return count
# # def rotate(n:int):
# #     num = [int(d) for d in str(n)]
# #     print("no",n)
# #     for i in range(len(num)):
# #         if num[i] == 2:
# #             num[i] = 5
# #         elif num[i] == 5:
# #             num[i] = 2
# #         elif num[i] == 6:
# #             num[i] = 9
# #         elif num[i] == 9:
# #             num[i] = 6
# #         elif num[i] == 0:
# #             continue
# #         elif num[i] == 8:
# #             continue
# #         elif num[i] == 1:
# #             continue
# #         else:
# #             return False
# #     newnum = 0
# #     for i in num:
# #         newnum=newnum*10
# #         newnum+=i
# #     print(newnum)
# #     if newnum!=n:
# #         return True
# #     return False
# #
# #
# # def maxDistaToClosest(seats):
# #     max0s = 0
# #     maxindex = 0
# #     currmax = 0
# #     start = False
# #     first0s = 0
# #     for i, x in enumerate(seats):
# #         if x == 1:
# #             start = True
# #             currmax = 0
# #         if x == 0:
# #             currmax += 1
# #             if start==False:
# #                 first0s += 1
# #         if currmax > max0s:
# #             max0s = currmax
# #             maxindex = i
# #         if currmax == max0s:
# #             if i == len(seats) - 1:
# #                 max0s = currmax
# #                 maxindex = i
# #     dis = 0
# #     print(first0s)
# #     print(maxindex, max0s)
# #     if len(seats) == maxindex + 1:
# #         dis = max0s
# #     elif max0s % 2 == 1:
# #         dis = int(max0s / 2) + 1
# #     elif max0s % 2 == 0:
# #         dis = int(max0s / 2)
# #     if first0s>dis:
# #         dis = first0s
# #     return dis
# # def maxDistToClosest(seats):
# #     N = len(seats)
# #     left, right = [N] * N, [N] * N
# #     for i in range(N):
# #         print(left)
# #         if seats[i] == 1: left[i] = 0
# #         elif i > 0: left[i] = left[i-1] + 1
# #     print(left)
# #     print()
# #     for i in range(N-1, -1, -1):
# #         print(right)
# #
# #         if seats[i] == 1: right[i] = 0
# #         elif i < N-1: right[i] = right[i+1] + 1
# #     print(right)
# #     for i, seat in enumerate(seats):
# #         if not seat:
# #             print(left[i], right[i])
# #             print(min(left[i],right[i]))
# #     return max(min(left[i], right[i])
# #                for i, seat in enumerate(seats) if not seat)
# # # print(maxDistToClosest([0,0,0,1]))
# # # nums = [2,2,1,1,1,2,2]
# # # dict = {}
# # # for x in set(nums):
# # #     dict[x] = 0
# # # for x in nums:
# # #     dict[x]+=1
# # # print(dict)
# # # a = [1,2,3,4,5]
# # # print(a[::-1])
# # matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# # mama = [[None]*len(matrix[0])]*len(matrix)
# # def maxsq(matrix):
# #     rows = len(matrix)
# #     cols = len(matrix[0]) if rows > 0 else 0
# #     dp = [[-1] * (cols + 1)] * (rows + 1)
# #     maxsqlen = 0
# #     for i in range(1, rows + 1):
# #         for j in range(1, cols + 1):
# #             if matrix[i - 1][j - 1] == "1":
# #                 dp[i][j] = min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1
# #                 maxsqlen = max(maxsqlen, dp[i][j])
# #     print(dp)
# #     return maxsqlen * maxsqlen
# # print(maxsq(matrix))
# # # print(longestCommonPrefix(["ditmemay","didmemay","ab"]))
# # # print("result",KMP("abaa", "aa"))
# # # print(computelps("aabaabaaa"))
# # # printNode(a)
# # list = ["abc", "cdi", "acd"]
# # print("-".join(list[::-1]))
# # x = False
# # x |= False
# # print(x)
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# # print(train_images[0].shape)
# # plt.imshow(train_images[5], cmap='gray')
# # plt.show()
# image = train_images[0].reshape((28, 28, 1))
# print(image.reshape((28,28)).shape)
# # coordinates = np.random.randint(0, 100, size=(30, 10, 2))
# # print(np.argmax(coordinates, axis=2).shape)

from random import randrange

def main():
    n = eval(input("How many rounds?"))
    list = []
    dcm = 0
    for i in range(n):
        dcm = sum()
        list.append(dcm)
    print(list)
    n = 8
    print(list.count(n))
def sum():
    dicesum = randrange(1,7) + randrange(1,7)
    return dicesum
# main()

class Animal:
    def __init__(self, loaivat):
        self.name = "cut"
        self.loai = loaivat
    def printdata(self):
        print(self.data)
    def themdata(self, x):
        self.data.append(x)

animal = Animal("dog")
animal1 = Animal("cat")
print(animal.name)
print(animal1.name)

def main():
    n = 30
    sumlist = []
    from random import randrange
    for i in range(n):
        sum = randrange(1, 7)+randrange(1,7)
        sumlist.append(sum)
    print(sumlist)
    max = 0
    for i in range(2, 13):
        count = sumlist.count(i)
        if count > max:
            max = count
    print(max)
import collections
nums = [1,1,1,3,4,5,6,6,5]
print(collections.Counter(nums).values())