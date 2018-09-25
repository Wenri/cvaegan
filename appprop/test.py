import numpy as np
from propDist import approxDist, fullDist

m=np.load('../output_cifar10_vgg_0/cvaegan/results/metric_epoch_0233.npz.npy')
attrs=np.load('../output_cifar10_vgg_0/cvaegan/results/datasets_attrs.npz.npy')

ret = fullDist(m[:10000],attrs[:10000], 0.4, 0.2)
#ret = approxDist(m[:4000],m[4000:5000],attrs[:5000],0.4)
acc = np.mean(np.argmax(ret[1000:10000], axis=1) == np.argmax(attrs[1000:10000], axis=1))

print(acc)
