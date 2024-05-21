"""Module for analysis of 2D nets in crystal structures

Functions
---------
csd : float
    Returns 'Combs Shift Distance'
densenets : dict
    Returns nets with max eff. density for given hkl
    (monkeypatched as Structure method)
flatten:

hklclust : list
    Returns clustering matrix for given hkl
    (monkeypatched as Structure method)
hklproj : list
    Returns projection of unit cell content on hkl vector
    (monkeypatched as Structure method)
med : float
    Returns mean effective density of nets
    (monkeypatched as Structure method)
netscif : None

netsearch :

readz : dict
    Reads clustering matrix
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


def densenets(self, indices):
    """Returns nets with max eff. density for given hkl

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k, l]

    Returns
    -------
    dict
        {'cluster': [[...], [...], [...], ...],
         'width': [w1, w2, w3, ...],
         'density': [d1, d2, d3, ...]}
        sorted from max eff. density down

    """

    N = len(self.p1().sites)
    d = dhkl(self.cell, indices)
    V = vol(self.cell)[0]

    x = readz(self.hklclust(indices))
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
    return result


Structure.densenets = densenets


def flatten(struct):
    """Translate sites along c axis to build contiguous nets

    Modifies input Structure instance!

    Parameters
    ----------
    struct : Structure

    Returns
    -------
    z_ave
        average z coordinate of resulting net
    """
    """
    z0 = struct.sites[0].fract[2]
    for s in struct.sites:
        z = s.fract[2]
        m = min(abs(z-z0), abs(z+1-z0), abs(z-1-z0))
        if m == abs(z-z0):
            continue
        elif m == abs(z+1-z0):
            s.fract[2] += 1
        else:
            Z[i] = z-1

    print(Z)
    """
    return


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

    if (indices == [0, 0, 0]) or (len(self.p1().sites) < 2):
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


def med(self, indices):
    """Returns mean effective density of nets

    Parameters
    ----------
    self : Structure
    indices : list
        [h, k ,l]

    Returns
    -------
    float
        mean eff. density
    """

    nets = self.densenets(indices)
    return (sum([len(c)*d for c, d in zip(nets['cluster'], nets['density'])])
            / len(self.p1().sites))


Structure.med = med


def netscif(struct, indices, path):
    # saves at 'path' basename cifs of individual nets
    # in spread basis
    n = 1
    for i in struct.nets(indices):
        name = path + f"_{indices[0]}{indices[1]}{indices[2]}_{n}.cif"
        struct.p1().sublatt(i[0]).transform(
            spreadbasis(indices, struct.cell)
        ).cif(name)
        n += 1


def netsearch(struct, groups=[], hklmax=[5, 5, 5]):
    # for each independent and coprime hkl (<= hklmax)
    # and sublattices listed in [groups]
    # returns [[hkl], smean, [smax]]
    # where smean is average effective density of nets
    # (averaged between sublattices), [smax] - max
    # eff density for each sublattice;
    # output sorted descending from max smean
    from math import gcd

    substr = []
    if groups == []:
        substr = [struct]
    else:
        for i in groups:
            substr.append(struct.sublatt(i))

    result = []
    for h in range(-hklmax[0], hklmax[0]+1):
        for k in range(-hklmax[1], hklmax[1]+1):
            for l in range(-hklmax[2], hklmax[2]+1):
                indices = [h, k, l]
                if gcd(*indices) != 1:
                    continue
                elif len([x for x in equivhkl(struct.symops,
                                              indices, laue=True)
                          if x in [y[0] for y in result]]) != 0:
                    continue
                else:
                    hklnets = []
                    smax = []
                    for i in substr:
                        inets = nets(i, indices)
                        hklnets += inets
                        smax.append(max([j[2] for j in inets]))
                    smean = (sum([len(i[0])*i[2] for i in hklnets])
                             / sum([len(i[0]) for i in hklnets]))
                    result.append([indices, smean, smax])
    result.sort(key=lambda x: x[1], reverse=True)
    return result


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

    result = {'cluster': [], 'width': []}
    for i in range(int(Z[-1][3])):
        result['cluster'].append([i])
        result['width'].append(0)
        #  listing singletons
    for i in range(len(Z)):
        n = readnode(Z, i)
        result['cluster'].append(n[0])
        result['width'].append(n[1])
    return result


def spreadbasis(indices, cell=[1, 1, 1, 90, 90, 90], m=10):
    # returns 4*4 P-matrix for transfer into
    # spread basis of hkl nets
    # (search within +-m unit cells)
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
