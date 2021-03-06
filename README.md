### SURE-CNN
 **SURE based Convolutional Neural Networks for Hyperspectral Image Denoising**


This paper addresses the hyperspectral image (HSI) denoising problem by using Stein's unbiased risk estimate (SURE) based convolutional neural network (CNN). Conventional deep learning denoising approaches often use supervised methods that minimize a mean-squared error (MSE) by training on noisy-clean image pairs. In contrast, our proposed CNN-based denoiser is unsupervised and only makes use of noisy images. The method uses SURE, which is an unbiased estimator of the MSE, that does not require any information about the clean image. Therefore minimization of the SURE loss function can accurately estimate the clean image only from noisy observation. Experimental results on both simulated and real hyperspectral datasets show that our proposed method outperforms competitive HSI denoising methods.

**Please cite our work if you are interested**

 @inproceedings{hvnguyen2020sure,
  title={{SURE} based Convolutional Neural Networks for Hyperspectral Image Denoising},
  author={Han Van, Nguyen and Magnus Orn, Ulfarsson and Johannes Runar, Sveinsson},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium},
  pages={},
  year={2020}
}

@inproceedings{nguyen2020sure,
  title={Sure based convolutional neural networks for hyperspectral image denoising},
  author={Nguyen, Han V and Ulfarsson, Magnus O and Sveinsson, Johannes R},
  booktitle={Proc. IEEE Geosci. Remote Sens. Symp},
  year={2020}
}
**Usage:**

Run the jupyter notebook file and see the results.


 - Data (preprocessing in Matlab) are in folder *his_data/Demo*
     + Data are the simulated noisy PU dataset with $\sigma=50/255$
     + Because of the limitation of the github space, if you need more data, please contact us.
 - CNN models are in folder *models*
 - Some helped functions are in the folder *utils*

Enviroment:

- Tensorflow 2.0 or higher
- Numpy
- Scipy, Skimage

