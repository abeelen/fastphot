import numpy as np
from scipy.signal import convolve2d, fftconvolve

def distribute_flux(image_shape, x, y, f):
    """Distribute fluxes between 4 adjacent pixels

    Parameters
    ----------
    image_shape : 2-tuple of int
        Shape of the output 2D image.
    x, y : floats or array_like
        the pixels coordinates of the sources
    f : corresponding flux(es)

    Returns
    -------
    data a :class:`numpy.ndarray` of shape `shape`

    Notes
    -----
    The input pixel(s) are distributed over the neighboring pixels
    with weight proportional to the intersection area.

    The pixels coordinates must be within the first and last pixels

    Examples
    --------

    >>> shape = (3, 3)
    >>> x, y = 0, 0 # Lower left pixel
    >>> data = distribute_flux(shape, x, y, 1)
    >>> data
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])

    >>> shape = (4, 4)
    >>> x, y = 1.5, 1.5 # Between four pixels
    >>> data = distribute_flux(shape, x, y, 1)
    >>> data
    array([[ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.25,  0.25,  0.  ],
           [ 0.  ,  0.25,  0.25,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ]])

    >>> shape = (4, 4)
    >>> x, y, f = [ 1, 1], [1, 1.5], [1, 2]
    >>> data = distribute_flux(shape, x, y, f)
    >>> data
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """


    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    fs = np.asarray(f, dtype=float)

    # If scalar inputs

    if xs.ndim == 0:
        xs = xs[None]
    if ys.ndim == 0:
        ys = ys[None]
    if fs.ndim == 0:
        fs = fs[None]

    assert np.all(xs >= 0) & np.all(xs <= image_shape[1]), "x must be within [0, image_shape[1]]"
    assert np.all(ys >= 0) & np.all(ys <= image_shape[0]), "y must be within [0, image_shape[0]]"
    assert len(xs) == len(ys) == len(fs), "x, y & f must have the same shape"

    fx = np.floor(xs)
    fy = np.floor(ys)
    rx = 1. - xs + fx
    ry = 1. - ys + fy
    sdatas = np.array([[ry * rx, (1 - ry) * rx],
                       [ry * (1 - rx), (1 - ry) * (1 - rx)]])

    cxs = fx.astype(int)
    cys = fy.astype(int)
    sdatas = sdatas * fs

    data = np.zeros(image_shape, dtype=float)
    for cx, cy, sdata in zip(cxs, cys, sdatas.T):
        data[cy:cy+2, cx:cx+2] += sdata

    return data

def make_ps_sources_image(image_shape, source_table, psf, oversamp=1):
    """Make an image containing point sources given a Point Spread Function.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Shape of the output 2D image.

    source_table : `~astropy.table.Table`
        Table of parameters for the point sources.  Each row of the
        table corresponds to a source whose parameters are
        defined by the column names.  The column names must include
        ``amplitude``, ``x_mean``, ``y_mean``.

    psf : `~numpy.ndarray`
        Point Spread function for the point sources

    Returns
    -------
    image : `~numpy.ndarray`
        Image containing point sources

    Notes
    -----
    FFT based method

    See Also
    --------
    photutils.make_gaussian_sources

    Examples
    --------

    .. plot::
        :include-source:

        # make a table of Gaussian sources
        from astropy.table import Table
        table = Table()
        table['amplitude'] = [50, 70, 150, 210]
        table['x_mean'] = [160, 25, 150, 90]
        table['y_mean'] = [70, 40, 25, 60]

        # output shape
        shape = (100, 200)

        # insure odd shape for kernel
        kernel_shape = 100 + 1
        kernel_1D = (np.arange(kernel_shape) - (kernel_shape-1)/2)**2 / (2 * 3**2)
        kernel = np.exp(-(kernel_1D + kernel_1D[:, None]))

        # point source image
        image_ps = make_ps_sources_image(shape, table, kernel)

        # make an image of the sources without noise
        from photutils.datasets import make_gaussian_sources_image

        # Need to define each gaussians
        table['x_stddev'] = [3, 3, 3, 3]
        table['y_stddev'] = [3, 3, 3, 3]
        table['theta'] = [0, 0, 0, 0]

        image_gs = make_gaussian_sources_image(shape, table)

    .. plot::
        :include-source:

        # output shape
        shape = (200, 200)

        # make a table of Gaussian sources
        from astropy.table import Table
        table = Table()
        n_sources = 10000
        table['amplitude'] = np.random.uniform(0,1, n_sources)
        table['x_mean'] = np.random.uniform(0, shape[1]-1, n_sources)
        table['y_mean'] = np.random.uniform(0, shape[0]-1, n_sources)


        # insure odd shape for kernel
        sigma = 3 # pixels
        table['x_stddev'] = np.ones(n_sources)*sigma
        table['y_stddev'] = np.ones(n_sources)*sigma
        table['theta'] = np.zeros(n_sources)
        kernel_shape = 200 + 1
        kernel_1D = (np.arange(kernel_shape) - (kernel_shape-1)/2)**2 / \
                    (2 * sigma**2)
        kernel = np.exp(-(kernel_1D + kernel_1D[:, None]))

        # point source image
        image_ps, image_ps_psf = make_ps_sources_image(shape, table, kernel)
        image_gs = make_gaussian_sources_image(shape, table)
        image_br = make_gaussian_ps_image_brut(shape, table)

        plt.close('all')
        _ = plt.hist(((image_ps_psf-image_gs)/image_gs).flatten()*100,
                     bins=500, range=(-1,1),alpha=0.5)
        _ = plt.hist(((image_ps-image_gs)/image_gs).flatten()*100,
                     bins=500, range=(-1,1),alpha=0.5)


    """

    data = distribute_flux(image_shape,
                           source_table['x_mean'],
                           source_table['y_mean'],
                           source_table['amplitude'])

    fft_freq = np.fft.fftfreq(psf.shape[0])
    fft_psf = np.abs(np.fft.fft2(psf))

    # Exact gaussian form...
    # sigma = 3
    # fft_psf = 2 * np.pi * 3**2 * np.exp(-2 * np.pi**2 * sigma**2 * (fft_freq**2 + fft_freq[:, None]**2))
    # back_fft = np.real(np.fft.fftshift(np.fft.ifft2(fft_psf)))

    # # For some reason, do not work as expected...
    size = 1
    fft_pix = np.sinc(fft_freq * size ) * \
              np.sinc(fft_freq * size  )[:, None]

    # Convolution of 2 pixels is a triangle function whom FT is a sinc**2
    # size = 1
    # fft_pix = np.sinc(fft_freq * size)**2 * \
    #           np.sinc(fft_freq * size)[:, None]**2


    kernel = np.real(np.fft.fftshift((np.fft.ifft2(fft_psf/fft_pix))))

    data = np.pad(data, kernel.shape, 'constant')

    data_ps =  fftconvolve(data, kernel, mode='same')
    data_psf = fftconvolve(data, psf, mode='same')

    yslice = slice(kernel.shape[1],-kernel.shape[1])
    xslice = slice(kernel.shape[0],-kernel.shape[0])
    data_ps = data_ps[yslice, xslice]
    data_psf = data_psf[yslice, xslice]

    return data_ps, data_psf

def make_gaussian_ps_image_center(image_shape, source_table, sigma=3):

    data = np.zeros(image_shape, dtype=np.float64)
    y, x = np.indices(image_shape)
    for source in source_table:
        data += source['amplitude'] * np.exp(-( (x-source['x_mean'])**2 +
                                                (y-source['y_mean'])**2) /
                                             (2 * sigma**2))
    return data

def make_gaussian_ps_image_integ(image_shape, source_table, sigma=3):
    from scipy.special import erf

    data = np.zeros(image_shape, dtype=np.float64)
    y, x = np.indices(image_shape)

    for source in source_table:
        data += source['amplitude'] * (2 * np.pi * sigma**2) / 4 * \
        (erf( (x+0.5-source['x_mean']) / (np.sqrt(2)*sigma) ) - \
         erf( (x-0.5-source['x_mean']) / (np.sqrt(2)*sigma) )) * \
        (erf( (y+0.5-source['y_mean']) / (np.sqrt(2)*sigma)) - \
         erf( (y-0.5-source['y_mean']) / (np.sqrt(2)*sigma)))

    return data

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    import matplotlib.pyplot as plt
    plt.ion()
    # output shape
    shape = (200, 200)

    # make a table of Gaussian sources
    from astropy.table import Table
    table = Table()
    n_sources = 1
    table['amplitude'] = np.random.uniform(0,1, n_sources) * 0 +1
    table['x_mean'] = np.random.uniform(0, shape[1]-1, n_sources) * 0 + 100.5
    table['y_mean'] = np.random.uniform(0, shape[0]-1, n_sources) * 0 + 100


    # insure odd shape for kernel
    sigma = 6 # pixels
    table['x_stddev'] = np.ones(n_sources)*sigma
    table['y_stddev'] = np.ones(n_sources)*sigma
    table['theta'] = np.zeros(n_sources)
    kernel_shape = 100 + 1
    kernel_1D = (np.arange(kernel_shape) - (kernel_shape-1)/2)**2 / \
                (2 * sigma**2)
    kernel = np.exp(-(kernel_1D + kernel_1D[:, None]))

    # point source image
    from photutils.datasets import make_gaussian_sources_image
    image_gs = make_gaussian_sources_image(shape, table)
    image_ps, image_ps_psf = make_ps_sources_image(shape, table, kernel)
    image_ce = make_gaussian_ps_image_center(shape, table, sigma=sigma)
    image_in = make_gaussian_ps_image_integ(shape, table, sigma=sigma)

    plt.close('all')
    fig, axes = plt.subplots(2, 4)

    axes[0,3].imshow(image_gs)
    axes[0,3].set_title('integ')

    im = axes[0,0].imshow(image_ps,
                          clim=axes[0,3].get_images()[0].get_clim())
    fig.colorbar(im, ax=axes[0,0], orientation='horizontal')
    axes[0,0].set_title('ps')

    im = axes[1,0].imshow(image_ps-image_in)
    fig.colorbar(im, ax=axes[1,0], orientation='horizontal')

    axes[0,1].imshow(image_ps_psf,
                     clim=axes[0,3].get_images()[0].get_clim())
    axes[1,1].imshow(image_ps_psf-image_in,
                     clim=axes[1,0].get_images()[0].get_clim())
    axes[0,1].set_title('psf')

    axes[0,2].imshow(image_gs,
                     clim=axes[0,3].get_images()[0].get_clim())
    axes[1,2].imshow(image_gs-image_in,
                     clim=axes[1,0].get_images()[0].get_clim())
    axes[0,2].set_title('gs')

    axes[1,3].hist(((image_gs-image_in)/image_in).flatten()*100,
                   bins=200, range=(-2,2),alpha=0.3, histtype='step')
    axes[1,3].hist(((image_ps_psf-image_in)/image_in).flatten()*100,
                   bins=200, range=(-2,2),alpha=0.5)
    axes[1,3].hist(((image_ps-image_in)/image_in).flatten()*100,
                   bins=200, range=(-2,2),alpha=0.5)

    for ax in axes[0:3, 0:3].flatten():
        ax.scatter(table['x_mean'], table['y_mean'], alpha=0.2)
        ax.set_xlim(0, image_ps.shape[1])
        ax.set_ylim(0, image_ps.shape[0])
