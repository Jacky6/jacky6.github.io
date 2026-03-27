# -*- coding: utf-8 -*-
'''
@Filename :    design_list_node.py
@Author   :    yixiu
@Email    :    jackyzheng9806@foxmail.com
@Time     :    2026-03-20 19:55:34
@Desc     :    
--------------------------------
'''

class ListNode:
    val: int
    next = None
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

