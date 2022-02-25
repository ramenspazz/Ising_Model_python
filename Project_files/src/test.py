# entry point
import Data_Analysis as DA
import timeit


class foo():
    def __init__(self):
        self.test_list = [1, 2, 3, 4, 5, 6]

    def add_link(self, link):
        self.test_list.append(link)

    def get_item(self):
        return(self.test_list[3])

    def __iter__(self):
        for link in self.test_list:
            yield link


TestCode1 = '''
def test1():
    testclass1 = foo()
    print(foo.get_item())
'''

TestCode2 = '''
def test2():
    testclass2 = foo()
    print(foo.test_list[3])
'''

times1 = []
times2 = []
for i in range(100000):
    times1.append(timeit.timeit(stmt=TestCode1, number=1000))
    times2.append(timeit.timeit(stmt=TestCode2, number=1000))
times1.sort()
mean1 = DA.data_mean(times1)
times2.sort()
mean2 = DA.data_mean(times2)
min1, max1, med1 = DA.ordinal_stats(times1)
min2, max2, med2 = DA.ordinal_stats(times2)
stdev1 = DA.std_dev(times1, mean1)
stdev2 = DA.std_dev(times2, mean2)
print(f'test1 min={min1}, max={max1}, median={med1}, mean={mean1}, stdev={stdev1}\n'
      f'test2 min={min2}, max={max2}, median={med2}, mean={mean2}, stdev={stdev2}\n')
