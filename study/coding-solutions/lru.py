class ListNode:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    """沿用原笔记约定：头部最旧，尾部最新。"""
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity 必须非负")
        self.capacity = capacity
        self.nodes = {}
        self.head, self.tail = ListNode(), ListNode()
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _append(self, node):
        node.prev, node.next = self.tail.prev, self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key):
        node = self.nodes.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._append(node)
        return node.val

    def put(self, key, value):
        if self.capacity == 0:
            return
        node = self.nodes.get(key)
        if node is not None:
            node.val = value
            self._remove(node)
        else:
            node = ListNode(key, value)
            self.nodes[key] = node
        self._append(node)
        if len(self.nodes) > self.capacity:
            old = self.head.next
            self._remove(old)
            del self.nodes[old.key]
