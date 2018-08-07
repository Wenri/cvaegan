TensorFlow cVAEGAN
===

TensorFlow implementation of cVAEGAN.

## Models

* Conditional variational autoencoder [Kingma et al. 2014]
* CVAE-GAN [Bao et al. 2017]

## Usage

### Prepare datasets

#### MNIST and SVHN

MNIST and SVHN datasets are automatically downloaded from their websites.

#### CelebA

First, download ``img_align_celeba.zip`` and ``list_attr_celeba.txt`` from CelebA [webpage](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Then, place these files to ``datasets`` and run ``create_database.py`` on ``databsets`` directory.

### Training

```shell
# Both standard and conditional models are available!
python train.py --model=cvaegan --epoch=200 --batchsize=100 --output=output
```

TensorBoard is also available with the following script.

```shell
tensorboard --logdir="output/dcgan/log"
```

### Results


#### CVAE-GAN (for SVHN 50 epochs)

<img src="results/svhn_cvaegan_epoch_0050_batch_73257.png" width="500px"/>

## References

* Kingma et al., "Auto-Encoding Variational Bayes", arXiv preprint 2013.
* Goodfellow et al., "Generative adversarial nets", NIPS 2014.
<!-- * Salimans et al., "Improved Techniques for Training GANs", arXiv preprint 2016. -->
<!-- * Zhao et al., "Energy-based generative adversarial network", arXiv preprint 2016. -->
<!-- * Dumoulin et al. "Adversarially learned inference", ICLR 2017. -->
* Kingma et al., "Semi-supervised learning with deep generative models", NIPS 2014.
* Bao et al., "CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training", arXiv preprint 2017.
