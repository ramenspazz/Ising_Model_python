# This file defines a queue made from a linked list
from __future__ import annotations
from multiprocessing import RLock
from threading import Lock
from time import time
from typing import Optional, TypeVar, Generic, TypedDict
import gc
gc.disable()
T = TypeVar('T')


class QueueEmpty(Exception):
    """
        Exception raised when LLQueue.pop is called and the Queue is empty.
    """
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message is not None:
            return(f'QueueEmpty, {self.message} ')
        else:
            return('QueueEmpty has been raised ')


class ItemAvailable(Exception):
    """
        Exception raised when LLQueue.pop is called and the Queue is empty,
        but item becomes available when waiting.
    """
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message is not None:
            return(f'ItemAvailable, {self.message} ')
        else:
            return('ItemAvailable has been raised ')


class LLNode(Generic[T]):
    def __init__(self, data, B_link, F_link):
        self.data: T = data
        self.forward_link: LLNode = F_link
        self.backward_link: LLNode = B_link

    def get_data(self) -> T:
        return(self.data)

    def get_fore_link(self) -> LLNode[T]:
        return(self.forward_link)

    def get_back_link(self) -> LLNode[T]:
        return(self.backward_link)

    def set_fore_link(self, fore_link) -> None:
        self.forward_link = fore_link

    def set_back_link(self, back_link) -> None:
        self.backward_link = back_link


class DictData(TypedDict):
    hash_val: int
    data: T


# The commented out lines are for the Wolff algorithm to run with spin
# flipping of entire groups of LIKE spins only. This was a test case and
# required me to check if a node was already in the queue or not.
# Left in just in case anyone else needs this functionality at a very small
# speed penelty. In my tests, calling IsInQueue only added (0.3 +/- 0.05)s per
# 1000 itterations of the Wolff algorithm
class LLQueue(Generic[T]):
    """
        A queue made from a linked list, operates on first in first out
        principal. Impliments locks to make `push`, `pop`, and `__len__`
        thread safe.
    """
    class_ref = None

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
        self.lock = Lock()
        self.rlock = RLock()

    def __iter__(self):
        return(self)

    def __next__(self):
        # with self.rlock:
        cur = self.head
        while cur is not None:
            yield(cur.get_data())
            cur = cur.get_fore_link()

    def __len__(self):
        with self.lock:
            return(self.size)

    def push(self, item) -> None:
        """
            Push a value to the end of the queue.
        """
        with self.lock:
            # queue is empty, create a new entry
            if self.size == 0:
                self.head = LLNode[T](item, None, None)
                self.size += 1
            # only one item in queue, update head and tail and make new node
            elif self.size == 1:
                self.tail = LLNode[T](item, self.head, None)
                self.head.set_fore_link(self.tail)
                self.size += 1
            # queue has more than one node, append new node and update tail
            else:
                new_node = LLNode[T](item, self.tail, None)
                self.tail.set_fore_link(new_node)
                self.tail = new_node
                self.size += 1
        # print(f'size of {self} is now {self.size}')

    def pop(self, timeout: Optional[float] = 0) -> T:
        """
            Remove an item from the front of the queue and return it. If the
            queue is empty and timeout is not set, raise `pyQueue`.`QueueEmpty`
            Exception, else return None if timeout is > 0.
        """
        with self.lock:
            try:
                if self.size == 0:
                    if timeout == 0:
                        raise QueueEmpty
                    else:
                        t_start = time()
                        now = time()
                        while now - t_start < timeout:
                            now = time()
                            if self.size > 0:
                                raise ItemAvailable
                        raise QueueEmpty
            except ItemAvailable:
                pass
            if self.size == 1:
                ret_val = self.head.get_data()
                del self.head
                self.head = None
                self.tail = None
                self.size -= 1
            else:
                ret_val = self.head.get_data()
                temp = self.head
                self.head = temp.get_fore_link()
                self.head.set_back_link(None)
                del temp
                self.size -= 1
            return(ret_val)

    def NextInQueue(self, item: LLNode[T]):
        with self.lock:
            return(item.forward_link.get_data())
