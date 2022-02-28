from audioop import mul
import multiprocessing
import threading as th
from threading import Event
from queue import Queue
from time import sleep


global_arr = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]


def foo(results: Queue, cv: th.Condition, finished):
    while not finished.is_set():
        temp = 0
        for val in global_arr:
            temp += val
        print(temp)
        results.put_nowait(temp)
        print('now waiting')
        cv.acquire(blocking=False)
        cv.wait()

tc = multiprocessing.cpu_count()
res = Queue(0)
cond1 = th.Event()
cond2 = th.Event()
thread_pool = []
ressum = 0
finished = False
for i in range(tc):
    thread_pool.append(th.Thread(
        target=foo,
        args=(res, cond1, cond2)))
    thread_pool[i].start()
while res.qsize() != tc:
    pass
cond2.set()
sleep(1)
cond1.set()
print('done')
for i in range(res.qsize()):
    ressum += res.get_nowait()
print(ressum)
res.queue.clear()
for t in thread_pool:
    t.join()
