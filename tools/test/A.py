import random
random.seed(10086)
def funcA():
    a = [random.randint(0, 9) for _ in range(3)]
    b = [random.randint(0, 9) for _ in range(3)]
    c = [random.randint(0, 9) for _ in range(3)]
    d = [random.randint(0, 9) for _ in range(3)]
    return