from mnist import load_mnist
import matplotlib.pyplot as plt
from experiment import run
from sklearn import svm

i, t = load_mnist()
"""
image = i[20]
imager = image.reshape((28,28))
plt.imshow(imager,cmap = "Greys")
plt.show()
"""
results = run(svm.SVC(gamma='auto'),
                 "grid_search",
                 {"kernel": ["linear", "poly", "rbf"], "degree": [3], "C": [0.1, 1, 10]},
                 i,
                 t)

print(results)