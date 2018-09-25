import numpy as np
from sklearn.neighbors import KNeighborsClassifier

m=np.load('../output_cifar10_vgg_0/cvaegan/results/metric_epoch_0233.npz.npy')
attrs=np.load('../output_cifar10_vgg_0/cvaegan/results/datasets_attrs.npz.npy')

radnn = KNeighborsClassifier()
radnn.fit(m[:1000], attrs[:1000])
lbls = radnn.predict(m[1000:10000])
acc = np.mean(np.argmax(lbls, axis=1) == np.argmax(attrs[1000:10000], axis=1))

print(acc)
