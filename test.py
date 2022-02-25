import numba
import time

# @numba.experimental.jitclass([('spin', numba.int8), ('links', numba.int64[:])])
class foo():
    def __init__(self):
        self.spin = 1
        # self.links = [1, 2, 3]

    def get_spin(self):
        return(self.spin)


class bar():
    def __init__(self):
        self.foobar = [foo()]*100

    def __iter__(self):
        for i in self.foobar:
            yield i

    def sum_arr(self):
        sum = 0
        for asdf in range(100000):
            for i in self.foobar:
                sum += i.spin
        return sum


# timing
baz = bar()
start = time.time()
baz.sum_arr()
end = time.time()
print(end - start)
