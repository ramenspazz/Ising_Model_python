import asyncio
import concurrent.futures

def blocking_io():
    # File operations (such as logging) can block the
    # event loop: run them in a thread pool.
    with open('/dev/urandom', 'rb') as f:
        return f.read(100)

def cpu_bound():
    # CPU-bound operations will block the event loop:
    # in general it is preferable to run them in a
    # process pool.
    return sum(i * i for i in range(10 ** 7))

async def main():
    loop = asyncio.get_running_loop()

    ## Options:

    # 1. Run in the default loop's executor:
    result = await loop.run_in_executor(
        None, blocking_io)
    print('default thread pool', result)

    # 2. Run in a custom thread pool:
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, blocking_io)
        print('custom thread pool', result)

    # 3. Run in a custom process pool:
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, cpu_bound)
        print('custom process pool', result)

asyncio.run(main())
# import multiprocessing as mltp
# from queue import Empty
# import MLTPQueue as queue
# from time import sleep


# global_arr = [1] * 32**2


# def foo(results: queue.MyQueue,
#         start_queue: queue.MyQueue,
#         start_itt,
#         wait_until_set,
#         finished):

#     while finished.is_set() is not True:
#         start_queue.put(1)
#         start_itt.wait()
#         temp = 0
#         for val in global_arr:
#             temp += val
#         results.put_nowait(temp)
#         wait_until_set.wait()
#     return


# qsize = mltp.cpu_count()
# res = queue.MyQueue()
# start_queue = queue.MyQueue()
# start_itt = mltp.Event()
# wait_until_set = mltp.Event()
# finished = mltp.Event()
# thread_pool = []
# ressum = 0

# for i in range(qsize):
#     thread_pool.append(mltp.Process(
#         target=foo,
#         args=(res,
#               start_queue,
#               start_itt,
#               wait_until_set,
#               finished)))
#     thread_pool[i].start()

# sleep(0.00001)
# for i in range(1000):
#     # wait_until_set.set()
#     wait_until_set.clear()
#     while start_queue.qsize() != qsize:
#         sleep(0.001)

#     try:
#         temp = 0
#         for j in range(qsize):
#             start_queue.get()
#     except Empty:
#         pass

#     start_itt.set()

#     while res.qsize() != qsize:
#         sleep(0.001)
#     ressum = 0
#     try:
#         for j in range(qsize):
#             ressum += res.get()
#     except Empty:
#         pass

#     start_itt.clear()
#     sleep(0.000001)
#     wait_until_set.set()

# finished.set()
# print('all finished')

# for t in thread_pool:
#     t.terminate()
