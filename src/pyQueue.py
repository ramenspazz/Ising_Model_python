# This file defines a queue made from a linked list
from __future__ import annotations
from typing import TypeVar, Generic
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


class LLQueue(Generic[T]):
    """
        A queue made from a linked list, operates on first in first out
        principal.
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def push(self, item) -> None:
        """
            Push a value to the end of the queue.
        """
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

    def pop(self) -> T:
        """
            Remove an item from the front of the queue and return it. If the
            queue is empty, raise `pyQueue`.`QueueEmpty` Exception.
        """
        if self.size == 0:
            raise QueueEmpty
        elif self.size == 1:
            ret_val = self.head.get_data()
            del self.head
            self.head = None
            self.tail = None
            self.size -= 1
            return(ret_val)
        else:
            ret_val = self.head.get_data()
            temp = self.head
            self.head = temp.get_fore_link()
            self.head.set_back_link(None)
            del temp
            self.size -= 1
            return(ret_val)
