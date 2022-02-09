import numpy as np

x = 40
y = 40
t = 16

MIN_INT = pow(-2, 31)
MAX_INT = pow(2, 31) - 1


def func(n, m, t: int) -> list:
    try:
        t = int(t)
        area = int(n * m)

        if not (MAX_INT > area > MIN_INT):
            raise ValueError('Input n and m are too large!')
        if t > area:
            return(0)
        elif t == area:
            return(1)

        test = int(0)
        div = int(1)
        div_overflow = int(0)
        OF_on = int(0)
        prev = int(0)

        while True:
            test = int(t * div_overflow) + int(t << div)

            if prev > area and test > area:
                return([int((t << div) / t + div_overflow),
                        area-t*np.floor(area / t)])

            if test == area:
                return([int((t << div) / t + div_overflow),
                        area-t*np.floor(area / t)])

            elif test < area:
                prev = test
                div += 1
                continue

            elif prev < area and OF_on == 0:
                div_overflow += int((t << (div - 1)) / t)
                prev = test
                # OF_on = 1
                div = 1
                continue
    except Exception:
        pass


Div, Rem = func(x, y, t)

print(Div, Rem)

prev = np.array([0, 0])
cur = np.array([0, 0])
for i in range(1, t+1, 1):
    tcord = [
        (i*Div) % x,
        int((i*Div) / x)
            ]
    print(f"{i} : from = {prev}, till = {tcord}")
    prev = tcord


def __NN_Worker__(self, bound: list, visited: list) -> int | Int:
    """
        Parameters
        ---------
        bounds : `list` | `ndarray` -> ArrayLike
            - bounds for NN summation.

        visited : `list`[`Node`]
            - `list` of already used nodes

        Returns
        -------
        spin_sum : `int` | `numpy.integer`
            - This threads final parital sum to return.
    """
    # Begin by creating a visit list for keeping track of nodes we still
    # need to visit and a visited list that will keep track of what has
    # already been visited.
    start_node = self[bound[0]]
    spin_sum: np.int256 = 0
    visit = list(start_node)
    cur = None
    while True:
        try:
            cur = visit.pop()
            if cur is not None and cur.get_spin() == 0:
                # pick a new node from visit
                visited.append(cur)
                continue
        except IndexError:
            # when the list is empty, break out of the while loop
            break
        except Exception:
            PE.PrintException()
        cur_neighbors = cur.get_connected()
        spin_sum += cur.get_spin()
        for nbr in cur_neighbors:
            if nbr not in visited:
                temp = nbr.get_spin()
                for item in nbr.get_connected():
                    visit.append(item)
                visited.append(nbr)
                spin_sum += temp
                continue
    return(spin_sum)