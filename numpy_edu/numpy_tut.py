import timeit
import numpy as np
import matplotlib.pyplot as plt

# numpy speed test
def python_forloop_list_approach(x, w):
    z = 0.
    for i in range(len(x)):
        z += x[i] * w[i]
    return z


a = [1., 2., 3.]
b = [4., 5., 6.]

print(python_forloop_list_approach(a, b))

large_a = list(range(1000))
large_b = list(range(1000))

print(python_forloop_list_approach(large_a, large_b))

exec_time = timeit.timeit('python_forloop_list_approach(large_a, large_b)', 
    globals=globals(), number=5)
print(f"Execution time: {exec_time} seconds")

def numpy_dotproduct_approach(x, w):
    # same as np.dot(x, w)
    # and same as x @ w
    return x.dot(w)
    

a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

print(numpy_dotproduct_approach(a, b))
np_large_a = np.arange(1000)
np_large_b = np.arange(1000)
exec_time2 = timeit.timeit('numpy_dotproduct_approach(np_large_a, np_large_b)', 
    globals=globals(), number=5)
print(f"Execution time (numppy): {exec_time2} seconds")

# calculate percentage improvement from exec_time to exec_time2
improvement = (exec_time - exec_time2) / exec_time * 100
print(f"Percentage improvement: {improvement:.2f}%")

a = [1., 2., 3.]
print(np.array(a))

lst = [[1, 2, 3], 
       [4, 5, 6]]
ary2d = np.array(lst)
print(ary2d)
print(ary2d.dtype)
print(ary2d.ndim)
ones = np.ones((3, 4, 6), dtype=np.int64)
print(ones)

eye = np.eye(3)
print(eye)

diag = np.diag((1, 2, 3))
print(diag)

step = np.arange(1., 11., 0.1)
print(step)
print(np.sqrt(step))
print(np.log(step))
print(np.add.reduce(step))


empty = np.empty((3, 4))
print(empty)
print(empty[:, :2])

np.random.seed(123)
rand = np.random.rand(3, 4)
print(rand)

# matplotlib
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()

rng = np.random.RandomState(123)
x = rng.normal(size=500)
y = rng.normal(size=500)
plt.scatter(x, y)
plt.xlabel('x-axis')
plt.ylabel('y-axis')

rng = np.random.RandomState(123)
x1 = rng.normal(0, 20, 1000) 
x2 = rng.normal(15, 10, 1000)

# fixed bin size
bins = np.arange(-100, 100, 5) # fixed bin size

plt.hist(x1, bins=bins, alpha=0.5)
plt.hist(x2, bins=bins, alpha=0.5)
plt.show()
