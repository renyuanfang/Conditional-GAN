# Conditional-GAN
This is an implementation of conditional GAN. We implement it using original CGAN, ACGAN, and ACGAN with residual blocks.
We test them on mnist, fashion-mnist and anime dataset.

# Mnist
CGAN            |  ACGAN     |  ACGAN_ResNet 
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/CGAN_mnist_epoch_50000_test.jpg)  |  ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_mnist_epoch_80000_test.jpg)  |  ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_ResNet_mnist_epoch_30000_test.jpg)

# Fashion-mnist
CGAN            |  ACGAN     |  ACGAN_ResNet 
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/CGAN_fashion-mnist_epoch_50000_test.jpg)  |  ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_fashion-mnist_epoch_80000_test.jpg)  |  ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_ResNet_fashion-mnist_epoch_30000_test.jpg)

# Anime
Here we do not include the test result of CGAN since it performs really worse on dataset with many labels.
We generate animes with blue hair, blue eye; blue hair, green eye; green hair, blue eye; green hair, red eye; pink hair, aqua eye; pink hair, purple eye; red hair, blue eye; red hair, brown eye. 

  ACGAN     |  ACGAN_ResNet 
:-------------------------:|:-------------------------:
 ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_anime_epoch_40000_test.jpg)  |  ![alt text](https://github.com/renyuanfang/Conditional-GAN/blob/master/images/ACGAN_ResNet_anime_epoch_40000_test.jpg)
