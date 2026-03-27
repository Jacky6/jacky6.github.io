# -*- coding: utf-8 -*-
'''
@Filename :    random_list_node.py
@Author   :    yixiu
@Email    :    jackyzheng9806@foxmail.com
@Time     :    2026-03-20 19:52:18
@Desc     :    复制带随机指针的链表
--------------------------------
'''
from typing import Optional

class RandomListNode:
    val:int
    next = None
    random = None

def copy_random_list_node(head: Optional[RandomListNode]) -> Optional[RandomListNode]:
    '''LeetCode138——复制带随机指针的链表'''
    if head == None:
        return None
    h = RandomListNode(-1)
    h.next = head
    p = h.next
    while p != None:   #建立双节点的单链表
        q = RandomListNode(p.val)
        q.next = p.next
        p.next = q
        p = q.next
    p = h.next
    while p != None:    #修改复制节点的random
        if p.random != None:
            p.next.random = p.random.next
        p = p.next.next
    r = h   # h 为复制单链表的头节点
    p = h.next
    while p != None:
        q = p.next
        p.next = q.next
        r.next = q
        r = q
        p = p.next
    r.next = None
    return h.next
