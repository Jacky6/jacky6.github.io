# -*- coding: utf-8 -*-
'''
@Filename :    list_node.py
@Author   :    yixiu
@Email    :    jackyzheng9806@foxmail.com
@Time     :    2026-03-20 19:46:36
@Desc     :    链表定义， 循环链表判断
--------------------------------
'''
from typing import Optional

class ListNode:
    val: int
    next = None
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    '''LeetCode 141: 循环链表'''
    if head is None:
        return False
    slow = head
    fast = head.next
    while fast is not None and fast.next is not None:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    return False

def test():
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = head
    print(has_cycle(head))

if __name__ == '__main__':
    test()