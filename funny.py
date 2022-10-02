# -*- coding: utf-8 -*-

"""Have Fun

Just for fun, not for research.
"""


def make24(x:list, N=24):
    # example:
    # make24([5,9,6,2], 24)
    #
    # (9+(5*(6/2)))

    L = len(x)
    if L == 0:
        raise ValueError('x is empty!')
    elif L == 1 and x[0] == N:
        return str(x[0])
    else:
        return ''

    for k in range(L):
        xk = x[k]
        xx = x.copy()
        xx.pop(k)
        yk = make24(xx, N-xk) # yk(xx)+xk = N
        if yk != '':
            return f'({xk}+{yk})'

        yk = make24(xx, N+xk) # yk(xx)-xk=N
        if yk != '':
            return f'({yk}-{xk})'

        yk = make24(xx, -N+xk) # xk-yk(xx)=N
        if yk != '':
            return f'({xk}-{yk})'

        if xk !=0:
            yk = make24(xx, N/xk) # yk(xx)*xk=N
            if yk != '':
                return f'({xk}*{yk})'

        yk = make24(xx, N*xk) # yk(xx)/xk=N
        if yk != '':
            return f'({yk}/{xk})'

        if N != 0:
            yk = make24(xx, xk/N) # xx/yk(xx)=N
            if yk != '':
                return f'({xk}/{yk})'

    return ''

