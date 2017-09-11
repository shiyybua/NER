def func():
    with open('resource/predict.txt', 'r') as f:
        for line in f.readlines():
            yield line
iter = func()
try:
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
    print next(iter)
except StopIteration:
    pass
