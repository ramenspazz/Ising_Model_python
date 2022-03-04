import multiprocessing as mltp
from MLTPQueue import ThQueue
from typing import Optional


class WaitListLock:
    """
        Defines a thread syncronization object that can wait and syncronisly
        start threads.
    """
    def __init__(self, thread_num: Optional[int] = 1):
        self.ListLock: list[ThQueue] = [ThQueue()] * thread_num
        self.WaitLock: list[ThQueue] = [ThQueue()] * thread_num
        self.start_threads = mltp.Event()

    def __getitem__(self, i: int) -> ThQueue:
        return(self.ListLock[i])

    def Wait(self, i: int) -> None:
        self.WaitLock[i].put_nowait(1)
        self.ListLock[i].get(block=True)

    def Check(self) -> None:
        for item in self.WaitLock:
            item.get(block=True)

    def Start_Threads(self) -> None:
        for q_item in self.ListLock:
            q_item.put_nowait(1)
