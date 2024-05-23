"""Module for analysis of 2D nets in crystal structures

Functions
---------
csd : float
    Returns 'Combs Shift Distance'
densenets : dict
    Returns nets with max eff. density for given hkl
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
    Returns max and mean effective density of nets
    (monkeypatched as Structure method)
readz : dict
    Reads clustering matrix
searchnets : dict
    Returns hkls of nets with max effective density
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
    """Returns nets with max eff. density for given hkl

    Parameters
    ----------
    self : Structure
    indices : list
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

    # import time

    # print(f'densenets started for {indices}')
    # starttime = time.time()
    N = len(self.p1().sites)
    d = dhkl(self.cell, indices)
    V = vol(self.cell)[0]

    x = readz(self.hklclust(indices))
    # print(f'readz completed in {time.time()-starttime:.2f} s')
    clusters = x['cluster']
    widths = x['width']
    densities = []
    for c, w in zip(clusters, widths):
        densities.append(
            len(c)*d/V * (1 - w/d*N / (len(c)-1)) if (len(c) > 1) else d/V)
        # implementation of effective density metrics
        # (for cluster of M sites equals M*d/V for zero width,
        # and 0 for width of d/N*(M-1))
    densities_new, clusters_new, widths_new = zip(
        *sorted(zip(densities, clusters, widths), reverse=True))
    result = {'cluster': [], 'width': [], 'density': []}
    clustered = []
    for c, w, d in zip(clusters_new, widths_new, densities_new):
        if len(set(c) & set(clustered)) == 0:
            result['cluster'].append(c)
            result['width'].append(w)
            result['density'].append(d)
            clustered += c
    if structures:
        result['structure'] = []
        P = spreadbasis(indices, self.cell)
        for i in result['cluster']:
            result['structure'].append(
                self.p1().sublatt(i).transform(P).flatten())
    # print(f'densenets completed in {time.time()-starttime:.2f} s')
    return result


Structure.densenets = densenets


def densenets2(self, indices, structures=True):
    """Returns nets with max eff. density for given hkl

    Parameters
    ----------
    self : Structure
    indices : list
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
    # import time

    # print(f'densenets started for {indices}')
    # starttime = time.time()
    N = len(self.p1().sites)
    d = dhkl(self.cell, indices)
    V = vol(self.cell)[0]

    x = readz2(self.hklclust(indices), d / N)
    # print(f'readz completed in {time.time()-starttime:.2f} s')
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
    # print(f'densenets completed in {time.time()-starttime:.2f} s')
    return result


Structure.densenets2 = densenets2


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


def mmed(self, indices):
    """Returns max and mean effective density of nets

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k ,l]

    Returns
    -------
    tuple
        (max, mean)
    """

    nets = self.densenets(indices, structures=False)
    return (max(nets['density']),
            sum([len(c)*d for c, d in zip(nets['cluster'], nets['density'])])
            / len(self.p1().sites))


Structure.mmed = mmed


def mmed2(self, indices):
    """Returns max and mean effective density of nets

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k ,l]

    Returns
    -------
    tuple
        (max, mean)
    """

    nets = self.densenets2(indices, structures=False)
    return (max(nets['density']),
            sum([len(c)*d for c, d in zip(nets['cluster'], nets['density'])])
            / len(self.p1().sites))


Structure.mmed2 = mmed2


def readz(Z):
    """Reads clustering matrix

    Parameters
    ----------
    Z : list
        clustering matrix

    Returns
    -------
    dict
        {'cluster': [[...], [...], [...], ...],
         'width': [w1, w2, w3, ...]}
    """

    # import time

    def readnode(Z, i):
        """Reads a node of Z-matrix

        Parameters
        ----------
        Z : list
            clustering matrix
        i : int
            index of node to read

        Returns
        -------
        tuple
            ([s1, s2, s3, ...], w) - list of singletons in the node
            and its width in single linkage metrics (sum of smallest
            M-1 distances between M singletons of the node)
        """

        N = int(Z[-1][3])  # number of singletons in Z
        singletons = []
        width = Z[i][2]
        for j in Z[i][:2]:
            if j < N:
                singletons.append(int(j))
            else:
                singletons += readnode(Z, int(j) - N)[0]
                width += readnode(Z, int(j) - N)[1]
        return (singletons, width)

    # print('readz started')
    # starttime = time.time()
    result = {'cluster': [], 'width': []}
    if len(Z) == 0:
        return {'cluster': [[0]], 'width': [0]}
    for i in range(int(Z[-1][3])):
        result['cluster'].append([i])
        result['width'].append(0)
        #  listing singletons
    for i in range(len(Z)):
        n = readnode(Z, i)
        # print(f'completed readnode {i}, elapsed {time.time()-starttime:.2f} s')
        result['cluster'].append(n[0])
        result['width'].append(n[1])
    # print(f'readz completed, elapsed {time.time()-starttime:.2f} s')
    return result


def readz2(Z, t):
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


def searchnets(self, groups=None, mode='mean', hklmax=[5, 5, 5]):
    """Returns hkls of nets with max effective density

    Parameters
    ----------
    self : Structure
    groups : list
        [[sitesA], [sitesB], ...] list of sublattices (default None)
    mode : str
        sorting of results (downward): 'mean' for mean eff. density,
        'max' for max eff. density (default 'mean')
    hklmax : list
        max abs of used h, k, l (default [5, 5, 5])

    Returns
    -------
    dict
        {'hkl': [[h1, k1, l1], [h2, k2, l2], ...],
         'max': [dmax1, dmax2, ...], 'mean': [dmean1, dmean2, ...]}
    """

    from math import gcd
    import time

    print(f'searchnets started with hklmax={hklmax}')
    starttime = time.time()

    substr = []
    if groups is None:
        substr = [self]
    else:
        for i in groups:
            if i != []:
                substr.append(self.sublatt(i))

    dhkls = []  # tuples (dhkl, [h, k, l])
    for h in range(hklmax[0], -hklmax[0]-1, -1):
        for k in range(hklmax[1], -hklmax[1]-1, -1):
            for l in range(hklmax[2], -hklmax[2]-1, -1):
                indices = [h, k, l]
                if gcd(*indices) != 1:
                    continue
                elif len([x for x in equivhkl(self.symops, indices, laue=True)
                          if x in [i[1] for i in dhkls]]) != 0:
                    continue
                else:
                    dhkls.append((dhkl(self.cell, indices), indices))
    print(f'prepared hkl list ({len(dhkls)} items, '
          f'{time.time()-starttime:.2f} s elapsed)')

    dhkls.sort(reverse=True)
    hkls = []
    maxs = []
    means = []
    dmin = 0.
    V = vol(self.cell)[0]
    counter = 1  # used only in messages
    for d, indices in dhkls:
        print(f'working with {indices} ({counter} of {len(dhkls)}, '
              f'{time.time()-starttime:.2f} s elapsed)\r', end='')
        counter += 1
        if d < dmin:
            continue
        submax = []
        submean = []
        for s in substr:
            mx, mn = s.mmed(indices)
            submax.append(mx)
            submean.append(mn)
        maxs.append(max(submax))
        Nsub = [len(s.p1().sites) for s in substr]
        mean = sum([d*N for d, N in zip(submean, Nsub)]) / sum(Nsub)
        means.append(mean)
        hkls.append(indices)
        if mean * V / sum(Nsub) > dmin:
            dmin = mean * V / sum(Nsub) > dmin
    if mode == 'max':
        maxs_new, hkls_new, means_new = zip(
            *sorted(zip(maxs, hkls, means), reverse=True))
    else:
        means_new, hkls_new, maxs_new = zip(
            *sorted(zip(means, hkls, maxs), reverse=True))
    print(f'searchnets completed, {time.time()-starttime:.2f} s elapsed)')
    return {'hkl': hkls_new, 'max': maxs_new, 'mean': means_new}


Structure.searchnets = searchnets


def searchnets2(self, groups=None, mode='mean', hklmax=[5, 5, 5]):
    """Returns hkls of nets with max effective density

    Parameters
    ----------
    self : Structure
    groups : list
        [[sitesA], [sitesB], ...] list of sublattices (default None)
    mode : str
        sorting of results (downward): 'mean' for mean eff. density,
        'max' for max eff. density (default 'mean')
    hklmax : list
        max abs of used h, k, l (default [5, 5, 5])

    Returns
    -------
    dict
        {'hkl': [[h1, k1, l1], [h2, k2, l2], ...],
         'max': [dmax1, dmax2, ...], 'mean': [dmean1, dmean2, ...]}
    """

    from math import gcd
    import time

    print(f'searchnets started with hklmax={hklmax}')
    starttime = time.time()

    substr = []
    if groups is None:
        substr = [self]
    else:
        for i in groups:
            if i != []:
                substr.append(self.sublatt(i))

    dhkls = []  # tuples (dhkl, [h, k, l])
    for h in range(hklmax[0], -hklmax[0]-1, -1):
        for k in range(hklmax[1], -hklmax[1]-1, -1):
            for l in range(hklmax[2], -hklmax[2]-1, -1):
                indices = [h, k, l]
                if gcd(*indices) != 1:
                    continue
                elif len([x for x in equivhkl(self.symops, indices, laue=True)
                          if x in [i[1] for i in dhkls]]) != 0:
                    continue
                else:
                    dhkls.append((dhkl(self.cell, indices), indices))
    print(f'prepared hkl list ({len(dhkls)} items, '
          f'{time.time()-starttime:.2f} s elapsed)')

    dhkls.sort(reverse=True)
    hkls = []
    maxs = []
    means = []
    dmin = 0.
    V = vol(self.cell)[0]
    counter = 1  # used only in messages
    for d, indices in dhkls:
        print(f'working with {indices} ({counter} of {len(dhkls)}, '
              f'{time.time()-starttime:.2f} s elapsed)\r', end='')
        counter += 1
        if d < dmin:
            continue
        submax = []
        submean = []
        for s in substr:
            mx, mn = s.mmed2(indices)
            submax.append(mx)
            submean.append(mn)
        maxs.append(max(submax))
        Nsub = [len(s.p1().sites) for s in substr]
        mean = sum([d*N for d, N in zip(submean, Nsub)]) / sum(Nsub)
        means.append(mean)
        hkls.append(indices)
        if mean * V / sum(Nsub) > dmin:
            dmin = mean * V / sum(Nsub) > dmin
    if mode == 'max':
        maxs_new, hkls_new, means_new = zip(
            *sorted(zip(maxs, hkls, means), reverse=True))
    else:
        means_new, hkls_new, maxs_new = zip(
            *sorted(zip(means, hkls, maxs), reverse=True))
    print(f'searchnets completed, {time.time()-starttime:.2f} s elapsed)')
    return {'hkl': hkls_new, 'max': maxs_new, 'mean': means_new}


Structure.searchnets2 = searchnets2


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
