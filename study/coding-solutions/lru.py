class ListNode:
    def __init__(self, key=0, val=0):
        self.val = val
        self.key = key
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hast_dict = dict()
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def move_to_tail(self, key):
        node = self.hast_dict[key]
        # 从当前位置摘出
        node.prev.next = node.next
        node.next.prev = node.prev
        # 插到 tail 前面
        node.prev = self.tail.prev
        node.next = self.tail
        # 要先写这个长的，不然先写 self.tail.prev = node的话，tail.prev已经改变了
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        node = self.hast_dict.get(key)
        if node:
            self.move_to_tail(node.key)
            return node.val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.hast_dict:
            self.hast_dict[key].val = value
            self.move_to_tail(key)
        else:
            if len(self.hast_dict) == self.capacity:
                # 弹出头部的节点
                self.hast_dict.pop(self.head.next.key)
                # 先将下一个节点更新
                self.head.next = self.head.next.next
                # 此时再将下一个节点指向head节点
                self.head.next.prev = self.head
                # 如果先 self.head.next.prev =  self.head 这就没意义，因为此时本来就是head

            self.hast_dict[key] = ListNode(key=key, val=value)
            node = self.hast_dict[key]
            node.prev = self.tail.prev
            node.next = self.tail
            self.tail.prev.next = node
            self.tail.prev = node
