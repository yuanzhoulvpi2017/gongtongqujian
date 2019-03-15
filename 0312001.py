import numpy as np 

#生成一个数据，100000行， 2维
x = np.random.randint(2, high=50, size=(10000000, 2))
x[:, 1] = x[:, 0] + 20


#从最大到最小，细分小点， 分多少，你可以自己定
y = np.linspace(x.min(), x.max(), 1000)

result = np.zeros((x.shape[0], len(y)), dtype=np.int8)
import sys
import time
print(sys.getsizeof(result))
total_time = time.time()
def f(i):
    l1 = np.array(x[:, 0] - y[i] <= 0 + 0)
    l2 = np.array(x[:, 1] - y[i] >= 0 + 0)
    return l1 * l2

result = np.array(np.array(list(map(f, range(len(y))))) + 0).T

print("all time: {:.1f}".format(time.time() - total_time))
print(np.max(np.sum(result, axis=0)))


