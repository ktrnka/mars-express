from __future__ import print_function
import timeit
import platform

import sys

import collections
import numpy


def get_mkl_version():
    try:
        import mkl
        return mkl.get_version_string()
    except ImportError:
        return getattr(numpy, "__mkl_version__", "none")


def benchmark_size(n, base_iterations=100, min_seconds_test=2., dtype="numpy.float64"):
    base_iterations = int(base_iterations)
    setup = "import numpy; import numpy.random; numpy.random.seed(4); a = numpy.random.rand({0}, {0}).astype({1}); b = numpy.random.rand({0}, {0}).astype({1})".format(n, dtype)

    # slower bench: ("numpy.linalg.svd(a)", 100), ("numpy.linalg.pinv(a)", 100), , ("numpy.subtract(a, b)", 100), ("numpy.tanh(a)", 100)
    results = collections.defaultdict(collections.defaultdict)
    for f, n_iter in [("numpy.dot(a, b)", base_iterations)]:
        time = timeit.timeit(f, setup=setup, number=n_iter)
        if time < min_seconds_test:
            per_sample = time / n_iter
            n_iter = int(min_seconds_test / per_sample + 0.5)
            time = timeit.timeit(f, setup=setup, number=n_iter)

        results[f]["unit_time"] = time / n_iter
        results[f]["n_iter"] = n_iter

        # print(f, n, 1000. * time / n_iter, n_iter)

    return results


print("Computer:", " | ".join(platform.uname()))
print("Interpreter:", sys.version)
print("Numpy: {}, MKL {}".format(numpy.__version__, get_mkl_version()))

for datatype in ["numpy.float32", "numpy.float64"]:
    print("Tests for ", datatype)

    # n -> function -> list of times
    allres = collections.defaultdict(list)
    for _ in range(5):
        for n in [100, 1000]:
            results = benchmark_size(n, base_iterations=(100 * 1000) / n, dtype=datatype)

            for f, res in results.items():
                allres["{0} @ {1}x{1}".format(f, n)].append(1000. * res["unit_time"])

    for test_name, times in allres.items():
        print("{test}: {min} ms  all={times}".format(test=test_name, min=min(times), times=times))
