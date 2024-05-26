"""Module for analysis of 2D nets in crystal structures

Functions
---------
csd : float
    Returns 'Combs Shift Distance'
densenets : dict
    Clusterizes atomic nets for given hkl
    (monkeypatched as Structure method)
flatten:
    Groups sites on a plane
    (monkeypatched as Structure method)
hklclust : list
    Returns clustering matrix for given hkl
    (monkeypatched as Structure method)
hklproj : list
    Returns projection of unit cell content on hkl vector
    (monkeypatched as Structure method)
mmed : float
    Returns mean and max effective density of nets
    (monkeypatched as Structure method)
readz : dict
    Reads clustering matrix
searchnets : dict
    Returns mean and max eff. density for hkls in range
    (monkeypatched as Structure method)
spreadbasis :
    Returns 'spread' basis for given Miller indices
"""

from core import dhkl, equivhkl, length, Structure, vol


def csd(x, y, d=1.):
    """Returns 'Combs Shift Distance'

    Parameters
    ----------
    x : float
        point from the first comb
    y : float
        point from the second comb
    d : float
        periodicity of combs

    Returns
    -------
    delta : float
        minimum distance between points of sets
        (x, x+-d, x+-2d, ...) and (y, y+-d, y+-2d, ...)
    """

    x = x % d
    y = y % d
    return min(abs(x - y), abs(x - y + d), abs(x - y - d))


def densenets(self, indices, structures=True):
    """Clusterizes atomic nets for given hkl

    Parameters
    ----------
    self : Structure
    indices : list or tuple
        [h, k, l]
    structures : bool
        whether create Structure for each cluster (default True)

    Returns
    -------
    dict
        {'cluster': [[...], [...], [...], ...],
         'width': [w1, w2, w3, ...],
         'density': [d1, d2, d3, ...]
         'structure': [Structure1, Structure2, Structure3, ...]}
        sorted from max eff. density down. When structures=True
        also creates list of Structure instances corresponding
        to each cluster in spread cell.
    """

    from scipy.spatial.distance import pdist

    N = len(self.p1().sites)
    d = dhkl(self.cell, indices)
    V = vol(self.cell)[0]

    x = readz(self.hklclust(indices), d / N)
    clusters = x['cluster']
    widths = []
    for i in clusters:
        proj = self.p1().sublatt(i).hklproj(indices)
        dist = pdist(proj,
                     metric=lambda x, y: csd(x, y, dhkl(self.cell, indices)))
        if len(dist) == 0:
            widths.append(0.)
        else:
            widths.append(max(dist))
    densities = []
    for c, w in zip(clusters, widths):
        densities.append(
            len(c)*d/V * (1 - w/d*N / (len(c)-1)) if (len(c) > 1) else d/V)
        # implementation of effective density metrics
        # (for cluster of M sites equals M*d/V for zero width,
        # and 0 for width of d/N*(M-1))
    densities_new, clusters_new, widths_new = zip(
        *sorted(zip(densities, clusters, widths), reverse=True))
    result = {'cluster': clusters_new, 'width': widths_new,
              'density': densities_new}
    if structures:
        result['structure'] = []
        P = spreadbasis(indices, self.cell)
        for i in result['cluster']:
            result['structure'].append(
                self.p1().sublatt(i).transform(P).flatten())
    return result


Structure.densenets = densenets


def flatten(self, ax=2):
    """Groups sites on a plane

    Modifies input Structure instance!

    Parameters
    ----------
    self : Structure
    ax : int
        axis for coordinate comparison (0: x, 1: y, 2: z)

    Returns
    -------
    Structure
    """

    if len(self.sites) < 2:
        return self
    x0 = self.sites[0].fract[ax] % 1
    for s in self.sites[1:]:
        s.fract[ax] = s.fract[ax] % 1
        x = s.fract[ax]
        for i in (x+1, x-1):
            if abs(i-x0) < abs(x-x0):
                x = i
        s.fract[ax] = x
    return self


Structure.flatten = flatten


def hklclust(self, indices):
    """Returns clustering matrix for given hkl

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k ,l]

    Returns
    -------
    list
        clustering matrix Z = [[node1, node2, distance, size], ...]
        sequentially describing merging of nodes
    """

    from scipy.cluster.hierarchy import linkage

    if len(self.p1().sites) == 1:
        return []
    elif len(self.p1().sites) == 0:
        return None
    return linkage(self.hklproj(indices), method='single',
                   metric=lambda x, y: csd(x, y, dhkl(self.cell, indices)))


Structure.hklclust = hklclust


def hklproj(self, indices):
    """Returns projection of unit cell content on hkl vector

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k ,l]

    Returns
    -------
    list
        [[x1], [x2], [x3], ...] (2D array for clusterization)
    """

    d = dhkl(self.cell, indices)
    return [[(sum([j.fract[i]*indices[i] for i in range(3)]) % 1) * d]
            for j in self.p1().sites]


Structure.hklproj = hklproj


def mnmxed(self, indices):
    """Returns mean and max effective density of nets

    Parameters
    ----------
    self : Structure
    indices : list or tuple
        [h, k ,l]

    Returns
    -------
    tuple
        (mean, max)
    """

    nets = self.densenets(indices, structures=False)
    return (sum([len(c)*d for c, d in zip(nets['cluster'], nets['density'])])
            / len(self.p1().sites), max(nets['density']))


Structure.mnmxed = mnmxed


def readz(Z, t):
    """Reads clustering matrix WITH THRESHOLD

    Parameters
    ----------
    Z : list
        clustering matrix
    t : float
        max distance between clusters

    Returns
    -------
    dict
        {'cluster': [[...], [...], [...], ...]}
    """

    from scipy.cluster.hierarchy import fcluster

    result = {'cluster': [], 'width': []}
    if len(Z) == 0:
        return {'cluster': [[0]], 'width': [0]}

    leaves = fcluster(Z, t, criterion='distance')
    nclust = max(leaves)
    for i in range(1, nclust+1):
        cluster = []
        for j in range(len(leaves)):
            if leaves[j] == i:
                cluster.append(j)
        result['cluster'].append(cluster)
    return result


def searchnets(self, filt=None, hklmax=[5, 5, 5]):
    """Returns mean and max eff. density for hkls in range

    Parameters
    ----------
    self : Structure
    filt : list
        [site1, site2, ...] list of sites to use, if None all will be used
        (default None)
    hklmax : list
        max abs of used h, k, l (default [5, 5, 5])

    Returns
    -------
    DataFrame
        {'hkl': [[h1, k1, l1], [h2, k2, l2], ...],
         'mean': [dmean1, dmean2, ...], 'max': [dmax1, dmax2, ...]}
    """

    from math import gcd
    from pandas import DataFrame
    import time

    print(f'searchnets started with hklmax={hklmax}')
    starttime = time.time()

    hkls = []
    for h in range(hklmax[0], -hklmax[0]-1, -1):
        for k in range(hklmax[1], -hklmax[1]-1, -1):
            for l in range(hklmax[2], -hklmax[2]-1, -1):
                indices = (h, k, l)
                if gcd(*indices) != 1:
                    continue
                elif len([x for x in equivhkl(self.symops, indices, laue=True)
                          if tuple(x) in hkls]) != 0:
                    continue
                else:
                    hkls.append(indices)
    print(f'prepared hkl list ({len(hkls)} items, '
          f'{time.time()-starttime:.2f} s elapsed)')

    means = []
    maxs = []
    counter = 1  # used only in messages
    for indices in hkls:
        print(f'working with {indices} ({counter} of {len(hkls)}, '
              f'{time.time()-starttime:.2f} s elapsed)\r', end='')
        counter += 1
        mn, mx = self.sublatt(filt).mnmxed(indices)
        means.append(mn)
        maxs.append(mx)
    print(f'\nsearchnets completed, {time.time()-starttime:.2f} s elapsed\n')
    return DataFrame({'hkl': hkls, 'mean': means, 'max': maxs})


Structure.searchnets = searchnets


def spreadbasis(indices, cell, m=10):
    """Returns 'spread' basis for given Miller indices

    Parameters
    ----------
    indices : list
        [h, k, l]
    cell : list
        [a, b, c, al, be, ga]
    m : int
        range of lattice points along a, b, and c for search
        of new basis vectors (default 10)

    Returns
    -------
    list
        4*4 P-matrix for basis transformation
    """

    from math import gcd
    from numpy.linalg import det

    h, k, l = indices
    ab = []
    c = []
    for x1 in range(-m, m+1):
        for x2 in range(-m, m+1):
            for x3 in range(-m, m+1):
                if gcd(x1, x2, x3) != 1:
                    continue
                elif (x1*h + x2*k + x3*l) == 0:
                    ab.append([x1, x2, x3])
                elif (x1*h + x2*k + x3*l) == 1:
                    c.append([x1, x2, x3])
    ab.sort(key=lambda x: length(cell, x))
    c.sort(key=lambda x: length(cell, x))
    P1 = ab[0]
    if ab[1] == [-i for i in ab[0]]:
        P2 = ab[2]
    else:
        P2 = ab[1]
    P3 = c[0]
    if det([P1, P2, P3]) < 0:
        P1, P2 = P2, P1
    return ([[P1[0], P2[0], P3[0], 0],
             [P1[1], P2[1], P3[1], 0],
             [P1[2], P2[2], P3[2], 0],
             [0, 0, 0, 1]])
