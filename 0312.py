import numpy as np 

#生成一个数据，100000行， 2维
x = np.random.randint(2, high=50, size=(1000000, 2))
x[:, 1] = x[:, 0] + 20


#从最大到最小，细分小点， 分多少，你可以自己定
y = np.linspace(x.min(), x.max(), 10000)

result = np.zeros((x.shape[0], len(y)), dtype=np.int8)
import sys
import time
sys.getsizeof(result)
total_time = time.time()
for i in range(len(y)):
    s_time = time.time()
    l1 = np.array(x[:, 0] - y[i] <= 0 + 0)
    l2 = np.array(x[:, 1] - y[i] >= 0 + 0)
    result[:, i] = l1 * l2
    print("epoch: {}, time loss: {:.2f}".format(i, time.time() - s_time))

print("all time: {:.1f}".format(time.time() - total_time))
print(np.max(np.sum(result, axis=0)))

