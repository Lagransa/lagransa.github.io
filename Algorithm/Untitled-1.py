import random
import collections
from collections import deque


# def reticulate_add(ls, count):
#     if len(ls) >= 1:
#         count += ls[0]
#         return reticulate_add(ls[1:], count)
#     else:
#         print( count)



# lst = [1, 7, 25, 10, 5, 6, 23]

# reticulate_add(lst, 0)\
# def quick_sort(ls):
#     baseline = 0
#     if len(ls) > 1:
#         small_ls = []
#         big_ls = []
#         for i in range(len(ls)):
#             if i != baseline:
#                 if ls[i] < ls[baseline]:
#                     small_ls.append(ls[i])
#                 elif ls[i] >= ls[baseline]:
#                     big_ls.append(ls[i])
#             else:
#                 continue
#         return quick_sort(small_ls) + [ls[baseline]] + quick_sort(big_ls)
#     else:
#         return(ls)


# def quick_sort(ls):
#     if len(ls) > 1:
#         baseline = random.randrange(len(ls)) #不知道怎么随机选择，回来再说
#         small_ls = [ls[i] for i in range(len(ls)) if (ls[i] <= ls[baseline] and i != baseline)]
#         big_ls = [ls[i] for i in range(len(ls)) if (ls[i] > ls[baseline] and i != baseline)]
#         return quick_sort(small_ls) + [ls[baseline]] + quick_sort(big_ls)
#     else:
#         return ls

# print(quick_sort(lst))


# graph = {}
# graph["you"] = ["Alice", "Bob", "Emilia"]
# graph["Alice"] = ["Shizuru"]
# graph["Bob"] = ["Sakano", "Simo"]
# graph["Emilia"] = ["Houkai"]
# graph["Shizuru"] = []
# graph["Sakano"] = []
# graph["Simo"] = []
# graph["Houkai"] = []


# checked = []


# def is_seller(name):
#     if name[-2:] == "no":
#         return True


# def search_seller(name):
#     search_queue = deque()  
#     search_queue += graph[name]
#     while search_queue:
#         person = search_queue.popleft()
#         if not person in checked:
#             if is_seller(person):
#                 print(f"Seller is {person}")
#                 return True
#             else:
#                 checked.append(person)
#     return False

# search_seller("you")

# states_needed = set(['mt', 'or', 'wa', 'id', 'nv', 'ut', 'ca', 'az'])
# stations = {}
# stations['kone'] = set(['id', 'nv', 'ut'])
# stations['ktwo'] = set(['id', 'wa', 'mt'])
# stations['kthree'] = set(['or', 'nv', 'ca'])
# stations['kfour'] = set(['nv', 'ut'])
# stations['kfive'] = set(['ca', 'az'])

# final_stations = set()

# def hungry_algorithm():
    
#     states_needed = set(['mt', 'or', 'wa', 'id', 'nv', 'ut', 'ca', 'az'])
#     stations = {}
#     stations['kone'] = set(['id', 'nv', 'ut'])
#     stations['ktwo'] = set(['id', 'wa', 'mt'])
#     stations['kthree'] = set(['or', 'nv', 'ca'])
#     stations['kfour'] = set(['nv', 'ut'])
#     stations['kfive'] = set(['ca', 'az'])

#     final_stations = set()

#     while states_needed:
#         best_station = None
#         state_covered = set()
#         for station, states_for_station in stations.items():
#             covered = states_for_station & states_needed
#             if len(covered) > len(state_covered):
#                 best_station = station
#                 state_covered = covered
#         final_stations.add(best_station)
#         states_needed -= state_covered

# hungry_algorithm()

# def longest_str(s):
#     if not s: return 0
#     lenth = len(s)
#     maximum = 0
#     curr_max = 0
#     left = 0
#     lookup = set()
#     for i in range(lenth):
#         curr_max += 1
#         while s[i] in lookup:
#             lookup.remove(s[left])
#             left += 1
#             curr_max -= 1
#         if curr_max > maximum:
#             maximum = curr_max
#         lookup.add(s[i])
#     return maximum

# def longest_str_optimized(s):
#     start = max_len = 0
#     used = {}  # 记录字符最后出现的位置
#     for i, c in enumerate(s):
#         if c in used and start <= used[c]:
#             start = used[c] + 1  # 直接跳跃到重复字符的下一位
#         max_len = max(max_len, i - start + 1)
#         used[c] = i
#     return max_len

# s = 'dvvdf'
# print(longest_str_optimized(s))

# def rotate(ls, k):
#     if k == 0 or k == len(ls):
#         return ls
#     return ls[-k:] + ls[:-k]

# tst = [0, 1, 2, 3, 4, 5, 6]
# print(rotate(tst, 4))

# def find_max_1(ls):
#     count = 0
#     fin = 0
#     for i in ls:
#         if i > 0:
#             count += 1
#         elif count > fin:
#             fin = count
#             count = 0
#         else:
#             count = 0
#     return fin

# ls = [1,1,0,1,1,1]
# print(find_max_1(ls))

# matrix = [[0] * 3 for _ in range(3)]
# x = 2
# y = 0
# while x >= 0:
#     matrix[x][y] = 1
#     x -= 1
#     y += 1

# tst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(tst * matrix)

# nums = [2,3,1,2,4,3]
# val = 7

# def minSubArrayLen(target, nums):
#     if target in nums:
#         return 1
#     if len(nums) == 0:
#         return 0
#     lenth = len(nums) + 1
#     count = 0
#     lft = 0
#     for i in range(len(nums)):
#         count += nums[i]
#         while count >= target:
#             if len(nums[lft:i]) + 1 < lenth:
#                 lenth = len(nums[lft:i]) + 1

#             count -= nums[lft]
#             lft += 1

                

    # if lenth == len(nums) + 1:
    #     return 0
    # else:
    #     return lenth

# def generateMatrix(n):
#     """
#     :type n: int
#     :rtype: List[List[int]]
#     """
#     lst = [[0] * n for _ in range(n)]
#     cycles = n // 2
#     count = 1

#     for cycle in range(1, cycles + 1):
#         for i in range(cycle - 1, n - cycle):
#             lst[cycle - 1][i] = count
#             count += 1
#         for j in range(cycle - 1, n - cycle):
#             lst[j][n - cycle] = count
#             count += 1
#         for a in range(n - cycle, cycle - 1, -1):
#             lst[n - cycle][a] = count
#             count += 1
#         for b in range(n - cycle, cycle - 1, -1):
#             lst[b][cycle - 1] = count
#             count += 1
#     if n % 2 == 1:
#         lst[cycles][cycles] = n ** 2
    
#     return lst

# minSubArrayLen(val, nums)
# generateMatrix(5)

class ListNode():
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
    def __repr__(self):
        next_val = self.next.val if self.next else None
        return f"ListNode[val={self.val}, next_val={next_val}]"


# class MyLinkedList(object):
#     def __init__(self):
#         self.head = None
#         self.size = 0
    
#     def __repr__(self):
#         vals = []
#         node = self.head
#         while node:
#             vals.append(str(node.val))
#             node = node.next
#         return f"MyLinkedList([{','.join(vals)}])"

#     def get(self, index):
#         """
#         :type index: int
#         :rtype: int
#         """
#         if index < 0 or index > self.size - 1:
#             return -1
#         fake_head = ListNode(None, self.head)
#         cur = fake_head
#         while index:
#             cur = cur.next
#             index -= 1
#         return cur.val
        

#     def addAtHead(self, val):
#         """
#         :type val: int
#         :rtype: None
#         """
#         self.head = ListNode(val, self.head)
#         self.size += 1
        
#     def addAtTail(self, val):
#         """
#         :type val: int
#         :rtype: None
#         """
#         fake_head = ListNode(None, self.head)
#         new_tail = ListNode(val)
#         cur = fake_head
#         while cur.next != None:
#             cur = cur.next
#         cur.next = new_tail
#         self.head = fake_head.next
#         self.size += 1

#     def addAtIndex(self, index, val):
#         """
#         :type index: int
#         :type val: int
#         :rtype: None
#         """
#         if index < 0 or index > self.size:
#             return -1
#         if index == self.size:
#             return self.addAtTail(val)
    
#         new_node = ListNode(val)
#         fake_head = ListNode(0, self.head)
#         cur = fake_head
#         while index:
#             cur = cur.next
#             index -= 1
#         new_node.next = cur.next #new_node(10,None)
#         cur.next = new_node #fakehead(0, 10)
#         self.head = fake_head.next
#         self.size += 1


#     def deleteAtIndex(self, index):
#         """
#         :type index: int
#         :rtype: None
#         """
#         if index < 0 or index > self.size - 1:
#             return -1
#         fake_head = ListNode(None, self.head)
#         cur = fake_head
#         while index:
#             cur = cur.next
#             index -= 1
#         cur.next = cur.next.next
#         self.head = fake_head.next
#         self.size -= 1
    
# tst = MyLinkedList()
# tst.addAtHead(4)
# tst.get(1)
# tst.addAtHead(1)
# tst.addAtHead(5)
# tst.deleteAtIndex(3)
# tst.addAtHead(7)
# tst.get(3)
# tst.get(3)
# tst.get(3)
# tst.addAtHead(1)
# tst.deleteAtIndex(4)
# tst.addAtIndex(0, 20)
# tst.addAtIndex(1, 30)
# tst.addAtTail(1)
# tst.get(0)

# class translinkedlist():
#     def reverseList(self, head):
#         if not head:
#             return
#         return self.reverse(head, None)

#     def reverse(self, cur, pre):
#         if not cur:
#             return pre
#         tmp = cur
#         cur = cur.next
#         tmp.next = pre
#         return self.reverse(cur, tmp)

# def trans_point(head):
#     if not head or not head.next:
#         return
#     cur = head
#     while cur:
#         tmp = cur.next
#         cur.next = cur.next.next
#         tmp.next = cur
#         cur = cur.next
#     return head

# tst = None
# for i in range(1, 6):
#     tst = ListNode(i ,tst)

# trans_point(tst)

# def removeNthFromEnd(head, n):
#     """
#     :type head: Optional[ListNode]
#     :type n: int
#     :rtype: Optional[ListNode]
#     """
#     if head.next == None:
#         return
#     fake_head = ListNode(0, head)
#     cur = fake_head
#     slower = fake_head
#     while n:
#         cur = cur.next
#     cur = cur.next
#     while cur:
#         cur = cur.next
#         slower = slower.next
#     slower.next = slower.next.next
#     return fake_head.next

# removeNthFromEnd(tst, 2)


# dic1 = {'A':3, 'B':2}
# dic2 = {'A':3, 'B':2}
# print(dic1 == dic2)


# def fourSumCount(nums1, nums2, nums3, nums4):
#     """
#     :type nums1: List[int]
#     :type nums2: List[int]
#     :type nums3: List[int]
#     :type nums4: List[int]
#     :rtype: int
#     """
#     dic = collections.Counter(a+b for a in nums1 for b in nums2)
#     count = 0
#     for c in nums3:
#         for d in nums4:
#             if -c-d in dic:
#                 count += dic[-c-d]
#     return count

    # if 1 == len(nums1) == len(nums2) == len(nums3) == len(nums4):
    #     return int(nums1[0] + nums2[0] + nums3[0] + nums4[0] == 0)
    # count = 0
    # lenth = len(nums1)
    # sum12 = {}
    # for i in range(lenth):
    #     for j in range(lenth):
    #         tmp = nums1[i] + nums2[j]
    #         sum12[tmp] = sum12.get(tmp, 0) + 1
    # for k in range(lenth):
    #     for l in range(lenth):
    #         tmp = nums3[k] + nums4[l]
    #         res = 0 - tmp
    #         if res in sum12:
    #             count += sum12[res]

    # return count


# num1 = [-1, -1]
# num2 = [1, -1]
# num3 = [-1, 1]
# num4 = [1, -1]

# fourSumCount(num1, num2, num3, num4)

# print(set(((-1, 1, 2), (-1, -1, 2), (-1, 1, 2))))


# def threesum(nums):
#     lenth = len(nums)
#     ls = []
#     nums.sort()
#     for i in range(lenth - 2):
#         if nums[i] > 0:
#             break
#         left = i + 1
#         right = lenth - 1
#         while right > left:        
#             rest = -nums[i]
#             tst = nums[left] + nums[right]
#             if tst > rest:
#                 right -= 1
#             elif tst < rest:
#                 left += 1
#             else:
#                 while nums[right] == nums[right - 1] & left < right:
#                     right -= 1
#                 while nums[left] == nums[left + 1] & left < right:
#                     left += 1
#                 ls.append([nums[i], nums[left], nums[right]])
#                 left += 1
#                 right -= 1
#     return ls

# tst1 = [1, 2, 0]
# print(tst1)



# def fourSum(nums, target):
#     lenth = len(nums)
#     nums.sort()
#     ls = []
#     for k in range(lenth - 3):
#         if nums[k] > 0 and nums[k] > target:
#             break
#         if nums[k] == nums[k - 1] and k > 0:
#             continue
#         for i in range(k + 1, lenth - 2):
#             if nums[i] == nums[i-1] and (i > k + 1):
#                 continue
#             left = i + 1
#             right = lenth - 1
#             base = nums[k] + nums[i]
#             res = target - base
#             while right > left:
#                 if nums[right] + nums[left] < res:
#                     left += 1
#                 elif nums[right] + nums[left] > res:
#                     right -= 1
#                 else:
#                     while nums[right] == nums[right - 1] and left < right:
#                         right -= 1
#                     while nums[left] == nums[left + 1] and left < right:
#                         left += 1
#                     ls.append([nums[k], nums[i], nums[left], nums[right]])
#                     right -= 1
#                     left += 1
#     return ls

# target1, target2 = 0, 8
# tst1 = [1,0,-1,0,-2,2]
# tst2 = [2, 2, 2, 2, 2]
# fourSum(tst1, target1)
# fourSum(tst2, target2)

# def reverseString(s):
#     """
#     :type s: List[str]
#     :rtype: None Do not return anything, modify s in-place instead.
#     """
#     tmp = ''
#     lenth = len(s)
#     odd = lenth % 2
#     step = lenth // 2
#     central = step + odd - 1
#     if odd:
#         for i in range(1, step + 1):
#             tmp = s[central + i]
#             s[central + i] = s[central - i]
#             s[central - i] = tmp
#     else:
#         for i in range(step):
#             tmp = s[central - i]
#             s[central - i] = s[central + i + 1]
#             s[central + i + 1] = tmp

    #正常解法
    # lenth = len(s)
    # l = 0
    # r = lenth - 1
    # while l < r:
    #     s[l], s[r] = s[r], s[l]
    #     r -= 1
    #     l += 1

# tst = ["H","a","n","n","a","h"]
# reverseString(tst)


# def reverseStr(s, k):
#     """
#     :type s: str
#     :type k: int
#     :rtype: str
#     """
#     lenth = len(s)
#     ls = list(s)
#     for i in range(0, lenth, k * 2):
#         res_len = len(ls[i:])
#         if res_len > k:
#             ls[i:i + k] = reverse(ls[i:i + k])
#         else:
#             ls[i:] = reverse(ls[i:])
#     return ''.join(ls)

# def reverse(s):
#     lenth = len(s)
#     l = 0
#     r = lenth - 1
#     while l < r:
#         s[l], s[r] = s[r], s[l]
#         l += 1
#         r -= 1
#     return s

# def reverseWords(s):
#     """
#     :type s: str
#     :rtype: str
#     """
#     lenth = len(s)
#     fast = 0
#     nw = ""
#     while fast < lenth:
#         if s[fast] != " ":
#             if nw != "":
#                 nw += " "
#             while fast < lenth and s[fast] != " ":
#                 nw += s[fast]
#                 fast += 1
#         else:
#             fast += 1
#     ls = list(nw)[::-1]
#     slow = 0
#     for i in range(len(ls)):
#         if ls[i] != ' ' and i != len(ls) - 1:
#             continue
#         elif i == len(ls) - 1:
#             ls[slow:] = ls[slow:][::-1]
#         else:
#             ls[slow:i] = ls[slow:i][::-1]
#             slow = i + 1

#     return ''.join(ls)

# string = 'all good   example'
# reverseWords(string)


# KMP：
# str: aabaababbaafaabaaf
# pattern: aabaaf
# j=0-4:成功，i=4
# j=5失配, i=5，π[5-1]=2，j=2，匹配成功，i++，j++，i=6，j=3
# j=3成功, i++，j++，i=7，j=4
# j=4失配，i=7，π[4-1]=1，j=1，失配，π[j-1]=0，i++，i=8，j=0
# i=8失配，j=0，i++，i=9
# i=9-10成功，i++++，j++++，i=11，j=2
# i=11失配，π[1]=0，i++，i=12，j=0
# i=12-17，j=0-5，匹配成功，返回，起始于18-6=12

# def repeatedSubstringPattern(s): #暴力1
#     """
#     :type s: str
#     :rtype: bool
#     """
#     if len(s) == 0:
#         return False
#     lenth = len(s)
#     pattern = ""
#     for i in range(lenth):
#         pattern += s[i]
#         len_2 = len(pattern)
#         if len_2 * 2 > lenth:
#             return False
#         if lenth % len_2 != 0:
#             continue
#         tmp = s
#         while tmp:
#             if tmp[:len_2] == pattern:
#                 tmp = tmp[len_2:]
#             else:
#                 break
#         if not tmp:
#             return True
#     return False

# def repeatedSubstringPattern(s): #暴力2
#     lenth = len(s)
#     if lenth <= 1:
#         return False
#     for i in range(0, lenth // 2 + 1):
#         if lenth % i != 0:
#             continue
#         pattern = s[:i]
#         if pattern * (lenth // i) == s:
#             return True
#     return False

# def repeatedSubstringPattern(s): #移动匹配法
#     lenth = len(s)
#     if lenth <= 1:
#         return False
#     tst = (s + s)[1:-1]
#     for i in range(len(tst) - lenth + 1):
#         if tst[i:i + lenth] == s:
#             return True
#     return False

# def repeatedSubstringPattern(s): #KMP法：
#     lenth = len(s)
#     if lenth <= 1:
#         return False
#     pi = [0] * lenth
#     j = 0
#     i = 1
#     while i < lenth:
#         if s[j] == s[i]:
#             pi[i] = j + 1
#             i += 1
#             j += 1
#         elif j != 0:
#             j = pi[j - 1]
#         else:
#             i += 1
    
#     if lenth % (lenth - pi[-1]) == 0 and pi[-1] != 0:
#         return True
#     return False


# s = "abcb"
# print(repeatedSubstringPattern(s))

# class stack_simulate_queue():
#     def __init__(ato):
#         ato.in_ls = []
#         ato.out_ls = []
    
#     def stack_in(ato, para):
#         ato.in_ls.append(para)

#     def stack_out(ato, para):
#         return ato.out_ls.pop(para)
    
#     def _in2out(ato):
#         if not ato.out_ls:
#             while ato.in_ls:
#                 ato.out_ls.append(ato.in_ls.pop())
    
#     def peek(ato):
#         ato._in2out()
#         return ato.out_ls[0]
    
#     def pop(ato):
#         ato._in2out()
#         return ato.out_ls.pop(0)

#     def is_empty(ato):
#         return ato.in_ls or ato.out_ls


# class queue_simulate_stack():
#     def __init__(self):
#         self.queue = []

#     def push(self, x):
#         self.queue.append(x)

#     def _trans(self, is_pop=True):
#         size = len(self.queue) - 1
#         while size:
#             self.push(self.queue.pop(0))
#             size -= 1
#         if not is_pop:
#             tmp = self.queue[0]
#             self.push(self.queue.pop(0))
#             return tmp

#     def pop(self):
#         self._trans()
#         return self.queue.pop(0)
    
#     def top(self):
#         return self._trans(False)
    
#     def is_empty(self):
#         return len(self.queue) == 0

# def isValid(s):
#     lenth  = len(s)
#     if lenth <= 1:
#         return False
#     stack = []
#     preparation = ['(', '[', '{']
#     excpetion_ = [')', ']', '}']
#     for i in range(lenth):
#         if s[i] in preparation:
#             ind = preparation.index(s[i])
#             stack.append(excpetion_[ind])
#         else:
#             if not stack:
#                 return False
#             tmp = stack.pop()
#             if tmp == s[i]:
#                 continue
#             else:
#                 return False
#     if stack:
#         return False
#     return True

# s = '([{)]}'

# def remove_depulicates(s): #新栈
#     n = len(s)
#     nw = []
#     count = 0
#     while n - count:
#         if s[count] not in nw or s[count] != nw[-1]:
#             nw.append(s[count])
#         else:
#             nw.pop()
#         count += 1
#     tmp = ''.join(nw)
#     return tmp
# def remove_neighbor(s): #字符串模拟栈
#     n = len(s)
#     nw = ""
#     count = 0
#     while n - count:
#         if not nw or s[count] != nw[-1]:
#             nw += s[count]
#         else:
#             nw = nw[:-1]
#         count += 1
#     return nw


# s = 'abbbabaaa'
# remove_depulicates(s)

# def evalRPN(tokens):
#     """
#     :type tokens: List[str]
#     :rtype: int
#     """
#     n = len(tokens)
#     result = []
#     operator = ['+', '-', '*', '/']
#     for i in range(n):
#         if tokens[i] not in operator:
#             result.append(int(tokens[i]))
#         else:
#             cal1 = result.pop()
#             cal2 = result.pop()
#             if tokens[i] == '+':
#                 result.append(cal1 + cal2)
#             elif tokens[i] == '-':
#                 result.append(cal2 - cal1)
#             elif tokens[i] == '*':
#                 result.append(cal2 * cal1)
#             else:
#                 if cal2 * cal1 > 0:
#                     result.append(int(cal2 / cal1))
#                 else:
#                     result.append(-int(abs(cal2) / abs(cal1)))
#     return result[0]

# tst = ["-78","-33","196","+","-19","-","115","+","-","-99","/","-18","8","*","-86","-","-","16","/","26","-14","-","-","47","-","101","-","163","*","143","-","0","-","171","+","120","*","-60","+","156","/","173","/","-24","11","+","21","/","*","44","*","180","70","-40","-","*","86","132","-84","+","*","-","38","/","/","21","28","/","+","83","/","-31","156","-","+","28","/","95","-","120","+","8","*","90","-","-94","*","-73","/","-62","/","93","*","196","-","-59","+","187","-","143","/","-79","-89","+","-"]
# print(evalRPN(tst))


# class My_queue(object):
#     def __init__(self):
#         self.queue = deque()
    
#     def pop(self, value):
#         if self.queue and self.queue[0] == value:
#             self.queue.popleft()
    
#     def push(self, value):
#         while self.queue and value > self.queue[-1]:
#             self.queue.pop()
#         self.queue.append(value)

#     def __getitem__(self, n):
#         return self.queue[n]


# def maxSlidingwindows(nums, k):
#     if len(nums) == 1 or k == 1:
#         return nums
#     result = []
#     queue = My_queue()

#     for i in range(k):
#         queue.push(nums[i])
#     result.append(queue[0])

#     for j in range(k, len(nums)):
#         queue.pop(nums[j - k])
#         queue.push(nums[j])
#         result.append(queue[0])

#     return result

# tst_nums, k = [1,3,1,2,0,5], 3
# print(maxSlidingwindows(tst_nums, k))

class TreeNode():
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# def preorder(root): #递归法
#     ls = []
#     def findNode(node):
#         if not node:
#             return
#         ls.append(node.val)
#         findNode(node.left)
#         findNode(node.right)
#     findNode(root)
#     return ls

# def preorder_iter(root): #迭代法
#     ls = []
#     stack = [root]
#     while stack:
#         tmp = stack.pop()
#         if tmp:
#             ls.append(tmp.val)
#             stack.append(tmp.right)
#             stack.append(tmp.left)
#     return ls


# def postorderTraversal(root): #迭代法，后序遍历
#     if not root:
#         return []
#     ls = []
#     stack = [root]
#     visited = []
#     while stack:
#         cur = stack[-1]
#         if cur.left and cur.left not in visited:
#             stack.append(cur.left)
#         elif cur.right and cur.right not in visited:
#             stack.append(cur.right)
#         else:
#             ls.append(cur.val)
#             visited.append(cur)
#             stack.pop()
#     return ls

# def inorderTraversal(root): #迭代法，中序遍历
#     if not root:
#         return []
#     ls = []
#     stack = [root]
#     visited = []
#     while stack:
#         cur = stack[-1]
#         if cur.left and cur.left not in visited:
#             stack.append(cur.left)
#         else:
#             visited.append(stack.pop())
#             ls.append(visited[-1].val)
#             if cur.right:
#                 stack.append(cur.right)
#     return ls

# def inorderTraversal(root): #迭代法教程版，中序遍历
#     ls = []
#     stack = []
#     cur = root
#     while cur or stack:
#         if cur:
#             stack.append(cur)
#             cur = cur.left
#         else:
#             cur = stack.pop()
#             ls.append(cur.val)
#             cur = cur.right
#     return ls

# def BFS(root):
#     if not root:
#         return []
#     ls = []
#     myque = deque()
#     myque.append(root)
#     cur = None
#     while myque:
#         tmp = []
#         size = len(myque)
#         while size:
#             cur = myque.popleft()
#             size -= 1
#             tmp.append(cur.val)
#             if cur.left:
#                 myque.append(cur.left)
#             if cur.right:
#                 myque.append(cur.right)
#         ls.append(tmp)
#     return ls
            
# def invert_treeNode_BFS(root): #层序遍历
#     if not root:
#         return root
#     cur = None
#     myque = deque()
#     myque.append(root)
#     while myque:
#         size = len(myque)
#         while size:
#             cur = myque.popleft()
#             if cur:
#                 cur.left, cur.right = cur.right, cur.left
#                 myque.append(cur.left)
#                 myque.append(cur.right)
#     return root

# def invert_treeNode_DFS_preorder(root): #递归，前序遍历
#     if not root:
#         return root
#     def invert(node):
#         if node:
#             node.left, node.right = node.right, node.left
#             invert(node.left)
#             invert(node.right)
#     invert(root)
#     return root




tree = TreeNode(5)
tree.left, tree.right = TreeNode(4), TreeNode(6)
tree.left.left, tree.left.right = TreeNode(2), TreeNode(1)
tree2 = TreeNode(1)
tree2.right = TreeNode(2)
tree2.right.left = TreeNode(3)
print(BFS(tree))