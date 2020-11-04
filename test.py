#!/usr/bin/env python3

def fun(x):
    if type(x) is int:
        print(x-1)
    else:
        print("{} - 1".format(str(x)))
