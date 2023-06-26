import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf


def GenNoise(X,sigma=25,eta=None):
    """Generate noise and adds to image
     argument: X is HSI with size (r,c,d)
               sigma: Noise level, i.e. std. dev. of Gaussian noise
               eta: defines how much the bell shape spread
     output: Noisy HSI and noise level"""
    if sigma>1:
        sigma_=sigma/255
    else:
        sigma_=sigma
    (r,c,d)=X.shape
    Xnoise=np.zeros((r,c,d))
    if eta is None:
        Xnoise=X+np.random.normal(0.,sigma_,(r,c,d))
        return Xnoise.astype(np.float32)
    else:
        den = 0
        num = []
        for i in range(d):
            den += np.exp(-np.square(i - d / 2) / (2 * np.square(eta)))
            num.append(np.exp(-np.square(i - d / 2) / (2 * np.square(eta))))
        temp = num / den
        band_sigma = np.sqrt(np.square(sigma_) * temp)  # noise level of each band
        for j in range(d):
            Xnoise[:, :, j] = X[:, :, j] + np.random.normal(0.0, band_sigma[j], (r, c))
        return Xnoise.astype(np.float32),np.transpose(band_sigma).astype(np.float32)

def snrCal(Xref,X):
    if len(Xref.shape)==3:
        Xref=Xref.reshape(Xref.shape[0]*Xref.shape[1]*Xref.shape[2],1)
        X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2],1)
        sig=np.sum(X**2)
        noi=np.sum((X-Xref)**2)
    else:
        pass
    return 10*np.log(sig/noi)/np.log(10)
def PSNR(img1, img2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    max_value=np.max(img1)
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

"""Define SURE loss function for inostropic Gaussian noise (sigma=const)"""
@tf.function
def loss(model,x,y,sigma,sure=True):#x is input, y is target
    loss_object = tf.keras.losses.MeanSquaredError()
    n=tf.dtypes.cast(tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3],tf.float32)
    batch=tf.dtypes.cast(tf.shape(x)[0],tf.float32)
    sigma_sq=tf.square(tf.reduce_mean(sigma))
    out=model(x)
    # mse=1/(n*batch)*(tf.reduce_sum(tf.square(y-out)))
    mse=loss_object(y_true=y,y_pred=out)
    e = 1e-3
    b = tf.random.normal((tf.shape(x)), 0.0, 1.0)
    in_pertubed = x + (b * e)
    out_pertubed = model(in_pertubed)
    z = out_pertubed - out
    dx = tf.reduce_sum(b * z)
    divMC = dx / e
    divterm = (2.0 * sigma_sq * divMC) / n
    sure_loss = mse - sigma_sq + divterm
    # sure_loss = mse + divterm

    if sure:
        return sure_loss, divterm
    else:
        return mse,divterm

"""Define SURE loss function for HSI with each band has different sigma"""
@tf.function
def losshyper(model,x,y,sigma,sure=True,e=1e-3):
    """SURE Loss function for hyperspectral image
    Input: x is input of network in shape [batch,M,N,B]
           y is output of network in shape [batch,M,N,B]
           sigma is noise level (std. dev) of each band in shape [1,B]
           model is network model
    Output: SURE loss value, if SURE is true, else MSE loss value, is scalar number.
            divterm: Network divergence is a scalar.
    """
    N=tf.dtypes.cast(tf.shape(x)[1]*tf.shape(x)[2],tf.float32)
    band=tf.dtypes.cast(tf.shape(x)[3],tf.float32)
    batch=tf.dtypes.cast(tf.shape(x)[0],tf.float32)
    sigma_sq=tf.square(tf.squeeze(sigma))

    b = tf.random.normal((tf.shape(x)), 0.0, 1.0)
    in_pertubed = x + (b * e)
    out=model(x)
    out_pertubed = model(in_pertubed)
    # Reshape to matrix
    OUT = tf.reshape(out,[tf.shape(out)[1]*tf.shape(out)[2],tf.shape(out)[3]])
    OUT_pertubed = tf.reshape(out_pertubed,[tf.shape(out_pertubed)[1]*tf.shape(out_pertubed)[2],tf.shape(out_pertubed)[3]])
    Y = tf.reshape(y,[tf.shape(y)[1]*tf.shape(y)[2],tf.shape(y)[3]])
    B = tf.reshape(b, [tf.shape(b)[1] * tf.shape(b)[2], tf.shape(b)[3]])
    # mse = (1/N)*tf.reduce_sum(tf.square(Y-OUT),0) # each band mse loss (old version)
    mse = tf.reduce_mean(tf.square(Y - OUT), 0)
    
    z = OUT_pertubed - OUT
    dx = tf.reduce_sum((B * z),0)
    divMC = dx / e
    divterm = (2.0 * sigma_sq * divMC) / N #each band
    sure_loss = mse + divterm - sigma_sq  # each band sureloss

    if sure:
        return  tf.reduce_mean(sure_loss), tf.reduce_mean(divterm)
    else:
        return  tf.reduce_mean(mse), tf.reduce_mean(divterm)
