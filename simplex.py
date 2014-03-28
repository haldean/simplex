try:
    import numpypy
except ImportError:
    pass
import numpy
import random
import sys

usage = '''
input format:
    max c1 c2 c3 ...
    restrict a11 a12 a13 ... b1
    restrict a21 a22 a23 ... b2
    restrict a31 a32 a33 ... b3

translates to:
    maximize c1 * x1 + c2 * x2 + c3 * x3
    under the constraints:
        a11 * x1 + a12 * x2 + a13 * x3 = b1
        a21 * x1 + a22 * x2 + a23 * x3 = b2
        a31 * x1 + a32 * x2 + a33 * x3 = b3
        \\forall xi, xi >= 0

all values in b must be nonnegative. note that one slack variable will be added
to each constraint.
'''

class BadFormatError(Exception):
    pass

class linprog(object):
    def __init__(self, lines):
        self.rows = len(lines) - 1

        funcline = lines[0]
        if not funcline.startswith('max '):
            raise BadFormatError()

        self.c = numpy.array(map(float, funcline.split(' ')[1:])).T
        # one slack var for every row, too
        self.cols = len(self.c) + self.rows

        # +1's are to compensate for initial column for Z
        self.basic_cols = set(range(len(self.c) + 1, self.cols + 1))

        avals = []
        bvals = []
        for line in lines[1:]:
            vals = map(float, line.split(' ')[1:])
            avals.extend(vals[:-1])
            bvals.append(vals[-1])
        self.A = numpy.array(avals).reshape(self.rows, len(self.c))
        self.b = numpy.array(bvals).T

        # table column indeces
        start_A = 1
        end_A = start_A + self.A.shape[1]
        start_I = end_A
        end_I = start_I + len(self.basic_cols)

        self.tab = numpy.zeros((self.rows + 1, self.cols + 2))
        self.tab[0, 0] = 1

        # add objective row
        self.tab[0, 1:len(self.c) + 1] = self.c.T

        # add constraint constants
        self.tab[1:, -1] = self.b.T

        # add nonbasic constraint coefficients
        self.tab[1:, start_A:end_A] = self.A

        # add basic constraint coefficients
        self.tab[1:, start_I:end_I] = numpy.identity(self.rows)

    def pivot(self, row, var):
        recip = 1 / self.tab[row, var]
        self.tab[row] *= recip
        for i in range(len(self.tab)):
            if i == row:
                continue
            vval = self.tab[i, var]
            self.tab[i] -= vval * self.tab[row]

        for col in self.basic_cols:
            if self.tab[0, col] != 0:
                self.basic_cols.remove(col)
                break
        self.basic_cols.add(var)

    def select_pivot(self):
        col = random.choice(list(self.nonbasic_cols()))
        x = self.tab[1:, col]
        r = self.tab[1:, -1]
        ratios = r / x
        min_i, min_v = 0, ratios[0]
        for i, v in enumerate(ratios):
            if (min_v < 0 or v < min_v) and v > 0:
                min_i, min_v = i, v

        # add the 1 here to compensate for dropping the first row in x and r
        return (min_i + 1, col)

    def nonbasic_cols(self):
        return set(range(1, self.cols + 1)) - self.basic_cols

    def iter_complete(self):
        return all(map(lambda x: x <= 0, self.tab[0, 1:]))

    def min_value(self):
        if not self.iter_complete():
            return None
        return self.tab[0, -1]

    def min_x(self):
        if not self.iter_complete():
            return None
        x = numpy.zeros(self.cols)
        for col in self.basic_cols:
            for i, v in enumerate(self.tab[1:, col]):
                if v:
                    x[col - 1] = self.tab[i + 1, -1]
        return x[:len(self.c)]

    def __str__(self):
        return 'A:\n%s\nb: %s\nc: %s\ntableau:\n%s' % (
                self.A, self.b.T, self.c.T, self.tab)
    __repr__ = __str__

def main():
    lines = sys.stdin.read().strip().split('\n')
    if not lines:
        print(usage)
        return
    try:
        lp = linprog(lines)
    except BadFormatError:
        print(usage)
        return

    i = 0
    iter_limit = 10 * len(lp.c.T)
    while not lp.iter_complete():
        i += 1
        if i >= iter_limit:
            print('%s\n---\nno solution in %d iterations' % (lp, i))
            return
        pivot = lp.select_pivot()
        lp.pivot(*pivot)

    print('found a solution!')
    print('minimum value = %f' % lp.min_value())
    print('         at x = %s' % lp.min_x());

if __name__ == '__main__':
    main()
