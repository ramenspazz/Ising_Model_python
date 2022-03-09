# This file defines a queue made from a linked list
from __future__ import annotations
from multiprocessing import RLock
from randomdict import RandomDict
from threading import Lock
from time import time
from typing import Optional, TypeVar, Generic, TypedDict

T = TypeVar('T')


class DictQueueEmpty(Exception):
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


class DictQueue(Generic[T]):
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
        self.member_dict = RandomDict()

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

    def clear(self):
        del self.head
        del self.tail
        del self.size
        del self.member_dict
        self.head = None
        self.tail = None
        self.size = 0
        self.member_dict = RandomDict()

    def push(self, item) -> None:
        """
            Push a value to the end of the queue.
        """
        with self.lock:
            self.member_dict[item] = item
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
                        raise DictQueueEmpty
                    else:
                        t_start = time()
                        now = time()
                        while now - t_start < timeout:
                            now = time()
                            if self.size > 0:
                                raise ItemAvailable
                        raise DictQueueEmpty
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
            del self.member_dict[ret_val]
            return(ret_val)

    def pop_rand(self, timeout: Optional[float] = 0) -> T:
        """
            Remove an item from the front of the queue and return it. If the
            queue is empty and timeout is not set, raise `pyQueue`.`QueueEmpty`
            Exception, else return None if timeout is > 0.
        """
        with self.lock:
            try:
                if self.size == 0:
                    if timeout == 0:
                        raise DictQueueEmpty
                    else:
                        t_start = time()
                        now = time()
                        while now - t_start < timeout:
                            now = time()
                            if self.size > 0:
                                raise ItemAvailable
                        raise DictQueueEmpty
            except ItemAvailable:
                pass
            if self.size == 1:
                ret_val = self.head.get_data()
                del self.head
                self.head = None
                self.tail = None
                self.size -= 1
            else:
                cur: LLNode = self.member_dict.random_value()
                ret_val = cur.get_data()
                prev_nd = cur.get_back_link()
                next_nd = cur.get_fore_link()
                if prev_nd is not None and next_nd is not None:
                    # selected node is between the head and tail
                    prev_nd.set_fore_link(next_nd)
                    next_nd.set_back_link(prev_nd)
                elif prev_nd is None:
                    # selected node is the head
                    pass
                elif next_nd is None:
                    # selected node is the tail
                    pass
                del cur
                self.size -= 1
            return(ret_val)

    def NextInQueue(self, item: LLNode[T]):
        with self.lock:
            return(item.forward_link.get_data())
