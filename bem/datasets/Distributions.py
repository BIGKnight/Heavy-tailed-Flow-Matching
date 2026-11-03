# file to generate some default data like swiss roll, GMM, levy variables

from sklearn.datasets import make_swiss_roll, make_moons, make_circles
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import scipy

def match_last_dims(data, size):
    """
    Expands a 1-dimensional tensor so that its last dimensions match the target size.
    
    Args:
        data (torch.Tensor): A 1-dimensional tensor with shape [batch_size].
        size (tuple or list): The target size, where size[0] should match batch_size.
    
    Returns:
        torch.Tensor: The expanded tensor with shape `size`.
    """
    # Ensure data is 1-dimensional
    assert data.dim() == 1, f"Data must be 1-dimensional, got {data.size()}"

    # Unsqueeze to add singleton dimensions for expansion
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    
    # Use expand instead of repeat to save memory
    return data.expand(*size).contiguous()

''' Generate fat tail distributions'''
# assumes it is a batch size
# is isotropic, just generates a single 'a' tensored to the right shape
def gen_skewed_levy(alpha, 
                    size, 
                    device = None, 
                    isotropic = True,
                    clamp_a = None):
    if (alpha > 2.0 or alpha <= 0.):
        raise Exception('Wrong value of alpha ({}) for skewed levy r.v generation'.format(alpha))
    if alpha == 2.0:
        ret = 2 * torch.ones(size)
        return ret if device is None else ret.to(device)
    # generates the alplha/2, 1, 0, 2*np.cos(np.pi*alpha/4)**(2/alpha)
    if isotropic:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size[0]), dtype=torch.float32)
        ret = match_last_dims(ret, size)
    else:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size), dtype=torch.float32)
    if clamp_a is not None:
        ret = torch.clamp(ret, 0., clamp_a)
    return ret if device is None else ret.to(device)


#symmetric alpha stable noise of scale 1
# can generate from totally skewed noise if provided
# assumes it is a batch size
def gen_sas(alpha, 
            size, 
            a = None, 
            device = None, 
            isotropic = True,
            clamp_eps = None):
    if a is None:
        a = gen_skewed_levy(alpha, size, device = device, isotropic = isotropic)
    ret = torch.randn(size=size, device=device)
    
    #if device is not None:
    #    ret = ret.to(device)
    #ret = repeat_shape(ret, size)
    ret = torch.sqrt(a)* ret
    if clamp_eps is not None:
        ret = torch.clamp(ret, -clamp_eps, clamp_eps)
    return ret

    '''
        if isotropic:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size[0]), dtype=torch.float32)
            ret = repeat_shape(ret, size)
        else:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size), dtype=torch.float32)
        return ret if device is None else ret.to(device)
    else:
        if isotropic:
            ret = torch.randn(size=(size[0],))
            ret = repeat_shape(ret, size)
        else:
            ret = torch.randn(size=size)
        if device is not None:
            ret = ret.to(device)
        return torch.sqrt(a)* ret'''


def _between_minus_1_1_with_quantile(x, quantile, scale_to_minus_1_1 = True):
    # assume x is centred
    high_quantile = torch.quantile(x, quantile, dim = 0, interpolation='nearest')
    low_quantile = torch.quantile(x, 1 - quantile, dim = 0, interpolation='nearest')
    assert not (high_quantile < 0).any()
    assert not (low_quantile > 0).any()
    clamp_value = torch.max(torch.abs(high_quantile), torch.abs(low_quantile))
    clamp_value = clamp_value.unsqueeze(0).repeat(x.shape[0], *tuple([1]* len(x.shape[1:])))
    tmp = torch.clamp(x, min= - clamp_value, max=clamp_value)
    idx_high = (tmp >= clamp_value)
    idx_low = (tmp <= - clamp_value)
    tmp[idx_high] = clamp_value[idx_high]
    tmp[idx_low] = -clamp_value[idx_low]
    if scale_to_minus_1_1:
        tmp /= clamp_value
    else:
        # just clamp
        tmp = tmp.clamp(-1, 1)
        #tmp = tmp
    return tmp




''' They must have the same signature'''

def sample_2_gmm(n_samples, 
                 alpha = None, 
                 n = None, 
                 std = None, 
                 theta = 1.0, 
                 weights = None, 
                 device = None, 
                 normalize=False, 
                 isotropic = False,
                 between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    if weights is None:
        weights = np.array([0.5, 0.5])
    means = np.array([ [theta, 0], [-theta, 0] ])
    gmm = GaussianMixture(n_components=2)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(2)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff, scale_to_minus_1_1=True) # should do something with 1 / sqrt(n)
    return x[torch.randperm(x.size()[0])]

def sample_grid_gmm(n_samples, 
                    alpha = None, 
                    n = None, 
                    std = None, 
                    theta = None, 
                    weights = None, 
                    device = None, 
                    normalize=False, 
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    means = []
    for i in range(n):
        for j in range(n):
            means.append([i, j])
    means = np.array(means)
    gmm = GaussianMixture(n_components=n*n)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(n*n)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    return x[torch.randperm(x.size()[0])]


def gen_swiss_roll(n_samples, 
                   alpha = None, 
                   n = None, 
                   std = None, 
                   theta = None, 
                   weights = None, 
                   device = None, 
                   normalize=False, 
                   isotropic = False,
                   between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=std)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    if between_minus_1_1:
        x = _between_minus_1_1_with_quantile(x, quantile_cutoff) # should do something with 1 / sqrt(n)
    return torch.tensor(x, dtype = torch.float32)


def sample_grid_sas(n_samples,
                    alpha = 1.8,
                    n = None,
                    std = None,
                    theta = 1.0,
                    weights = None,
                    device = None,
                    normalize=False,
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 0.99,
                    grid_spacing = 1.0):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    data = std * gen_sas(alpha, size = (n_samples, 2), isotropic = isotropic)
    weights = np.concatenate((np.array([0.0]), weights))
    idx = np.cumsum(weights)*n_samples
    for i in range(n):
        for j in range(n):
            # for the moment just selecting exact proportions
            s = int(idx[i*n + j])
            e = int(idx[i*n + j + 1])
            data[s:e] = data[s:e] + torch.tensor([i * grid_spacing, j * grid_spacing])

    # center the data such that the center of the grid is at 0
    data = data - torch.tensor([(n / 2) * grid_spacing, (n / 2) * grid_spacing])
    

    if normalize:
        data = (data - data.mean()) / data.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    #data = torch.tensor(data, dtype = torch.float32)
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff) # should do something with 1 / sqrt(n)
    return data[torch.randperm(data.size()[0])]


def sample_sas(n_samples = None,
               alpha = 1.8,
               n = None,
               std = None,
               theta = 1.0,
               weights = None,
               device = None,
               normalize=False,
               isotropic = False,
               between_minus_1_1 = False,
               quantile_cutoff = 0.99,
               **kwargs):
    clamp_eps = kwargs.get('clamp_eps', None)
    clamp_a = kwargs.get('clamp_a', None)
    a = kwargs.get('a', None)
    size = kwargs.get('size', None)
    if size is None:
        assert n_samples is not None
        size = (n_samples, 2)
        dataset_mode = True
    else:
        dataset_mode = False
    if a is None and clamp_a is not None:
        a = gen_skewed_levy(alpha,
                            size=size,
                            device=device,
                            isotropic=isotropic,
                            clamp_a=clamp_a)
    data = gen_sas(alpha,
                   size=size,
                   a=a,
                   device=device,
                   isotropic=isotropic,
                   clamp_eps=clamp_eps)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert not torch.isnan(data).any()
    if dataset_mode:
        assert data.shape == (n_samples, 2)
        data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_eight_gaussians(n_samples,
                           alpha = None,
                           n = None,
                           std = None,
                           theta = 2.0,
                           weights = None,
                           device = None,
                           normalize=False,
                           isotropic = False,
                           between_minus_1_1 = False,
                           quantile_cutoff = 0.99):
    if std is None:
        std = 0.02
    if weights is None:
        weights = np.array([1 / 8 for _ in range(8)])
    else:
        # Ensure weights has exactly 8 elements for 8 Gaussians
        weights = np.array(weights)
        if len(weights) != 8:
            weights = np.array([1 / 8 for _ in range(8)])
    means = theta * np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)],
    ])
    gmm = GaussianMixture(n_components=8)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for _ in range(8)]
    data, _ = gmm.sample(n_samples)
    data = torch.tensor(data, dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_two_moons(n_samples,
                     alpha = None,
                     n = None,
                     std = None,
                     theta = None,
                     weights = None,
                     device = None,
                     normalize=False,
                     isotropic = False,
                     between_minus_1_1 = False,
                     quantile_cutoff = 0.99):
    if std is None:
        std = 0.05
    data, _ = make_moons(n_samples=n_samples, noise=std)
    data = torch.tensor(data, dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_circles(n_samples,
                   alpha = None,
                   n = None,
                   std = None,
                   theta = None,
                   weights = None,
                   device = None,
                   normalize=False,
                   isotropic = False,
                   between_minus_1_1 = False,
                   quantile_cutoff = 0.99):
    if std is None:
        std = 0.05
    # factor must be in [0, 1); ignore theta if it's outside this range
    if theta is None or theta < 0 or theta >= 1:
        factor = 0.5
    else:
        factor = theta
    data, _ = make_circles(n_samples=n_samples, noise=std, factor=factor)
    data = torch.tensor(data, dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_two_spirals(n_samples,
                       alpha = None,
                       n = None,
                       std = None,
                       theta = 1.0,
                       weights = None,
                       device = None,
                       normalize=False,
                       isotropic = False,
                       between_minus_1_1 = False,
                       quantile_cutoff = 0.99):
    if std is None:
        std = 0.02
    n_half = (n_samples + 1) // 2
    theta_vals = np.random.rand(n_half) * 3.0 * np.pi
    r = theta_vals / (3.0 * np.pi)
    first = np.stack([r * np.cos(theta_vals), r * np.sin(theta_vals)], axis=1)
    second = np.stack([-r * np.cos(theta_vals), -r * np.sin(theta_vals)], axis=1)
    data = np.concatenate([first, second], axis=0)
    data = data[:n_samples] + np.random.randn(n_samples, 2) * std
    data = torch.tensor(data, dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_checkerboard(n_samples,
                        alpha = None,
                        n = None,
                        std = None,
                        theta = 1.0,
                        weights = None,
                        device = None,
                        normalize=False,
                        isotropic = False,
                        between_minus_1_1 = False,
                        quantile_cutoff = 0.99):
    x1 = np.random.rand(n_samples) * 4.0 - 2.0
    x2 = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2.0
    x2 = x2 + (np.floor(x1) % 2)
    data = np.stack([x1, x2], axis=1) * theta
    data = torch.tensor(data, dtype=torch.float32)
    if std is not None and std > 0:
        data = data + std * torch.randn_like(data)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_rings(n_samples,
                 alpha = None,
                 n = None,
                 std = None,
                 theta = 1.0,
                 weights = None,
                 device = None,
                 normalize=False,
                 isotropic = False,
                 between_minus_1_1 = False,
                 quantile_cutoff = 0.99):
    if std is None:
        std = 0.08
    if n is None:
        n = 3
    if n == 3:
        radii = np.array([0.5, 1.0, 1.5]) * theta
    else:
        radii = np.linspace(0.5, 1.5, n) * theta
    if weights is None:
        weights = np.array([1 / n for _ in range(n)])
    counts = np.random.multinomial(n_samples, weights / np.sum(weights))
    points = []
    for ring_id, ring_n in enumerate(counts):
        if ring_n == 0:
            continue
        angles = 2.0 * np.pi * np.random.rand(ring_n)
        radius = radii[ring_id] + std * np.random.randn(ring_n)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points.append(np.stack([x, y], axis=1))
    data = np.concatenate(points, axis=0) if len(points) > 0 else np.zeros((n_samples, 2))
    if data.shape[0] < n_samples:
        data = np.concatenate([data, np.zeros((n_samples - data.shape[0], 2))], axis=0)
    data = torch.tensor(data[:n_samples], dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def sample_olympic_rings(n_samples,
                         alpha = None,
                         n = None,
                         std = None,
                         theta = 1.0,
                         weights = None,
                         device = None,
                         normalize=False,
                         isotropic = False,
                         between_minus_1_1 = False,
                         quantile_cutoff = 0.99):
    if std is None:
        std = 0.04
    centers = theta * np.array([
        [-1.0, 0.5],
        [0.0, 0.5],
        [1.0, 0.5],
        [-0.5, 0.0],
        [0.5, 0.0],
    ])
    n_rings = centers.shape[0]
    if weights is None:
        weights = np.array([1 / n_rings for _ in range(n_rings)])
    else:
        # Ensure weights has exactly 5 elements for 5 rings
        weights = np.array(weights)
        if len(weights) != n_rings:
            weights = np.array([1 / n_rings for _ in range(n_rings)])
    counts = np.random.multinomial(n_samples, weights / np.sum(weights))
    radius = 0.4 * theta
    points = []
    for ring_id, ring_n in enumerate(counts):
        if ring_n == 0:
            continue
        angles = 2.0 * np.pi * np.random.rand(ring_n)
        radial_noise = std * np.random.randn(ring_n)
        ring_radius = radius + radial_noise
        x = centers[ring_id, 0] + ring_radius * np.cos(angles)
        y = centers[ring_id, 1] + ring_radius * np.sin(angles)
        points.append(np.stack([x, y], axis=1))
    data = np.concatenate(points, axis=0) if len(points) > 0 else np.zeros((n_samples, 2))
    if data.shape[0] < n_samples:
        data = np.concatenate([data, np.zeros((n_samples - data.shape[0], 2))], axis=0)
    data = torch.tensor(data[:n_samples], dtype=torch.float32)
    if normalize:
        data = (data - data.mean()) / data.std()
    if between_minus_1_1:
        data = _between_minus_1_1_with_quantile(data, quantile_cutoff)
    assert data.shape == (n_samples, 2)
    assert not torch.isnan(data).any()
    data = data[torch.randperm(data.size()[0])]
    return data if device is None else data.to(device)


def gen_two_moons(n_samples,
                  alpha = None,
                  n = None,
                  std = None,
                  theta = None,
                  weights = None,
                  device = None,
                  normalize=False,
                  isotropic = False,
                  between_minus_1_1 = False,
                  quantile_cutoff = 0.99):
    return sample_two_moons(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)


def gen_circles(n_samples,
                alpha = None,
                n = None,
                std = None,
                theta = None,
                weights = None,
                device = None,
                normalize=False,
                isotropic = False,
                between_minus_1_1 = False,
                quantile_cutoff = 0.99):
    return sample_circles(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)


def gen_two_spirals(n_samples,
                    alpha = None,
                    n = None,
                    std = None,
                    theta = 1.0,
                    weights = None,
                    device = None,
                    normalize=False,
                    isotropic = False,
                    between_minus_1_1 = False,
                    quantile_cutoff = 0.99):
    return sample_two_spirals(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)


def gen_checkerboard(n_samples,
                     alpha = None,
                     n = None,
                     std = None,
                     theta = 1.0,
                     weights = None,
                     device = None,
                     normalize=False,
                     isotropic = False,
                     between_minus_1_1 = False,
                     quantile_cutoff = 0.99):
    return sample_checkerboard(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)


def gen_rings(n_samples,
              alpha = None,
              n = None,
              std = None,
              theta = 1.0,
              weights = None,
              device = None,
              normalize=False,
              isotropic = False,
              between_minus_1_1 = False,
              quantile_cutoff = 0.99):
    return sample_rings(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)


def gen_olympic_rings(n_samples,
                      alpha = None,
                      n = None,
                      std = None,
                      theta = 1.0,
                      weights = None,
                      device = None,
                      normalize=False,
                      isotropic = False,
                      between_minus_1_1 = False,
                      quantile_cutoff = 0.99):
    return sample_olympic_rings(n_samples, alpha, n, std, theta, weights, device, normalize, isotropic, between_minus_1_1, quantile_cutoff)
