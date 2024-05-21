"""Module for analysis of 2D nets in crystal structures

Functions
---------
flatten:

nets :

netscif : None

netsearch :

netsearchf1 :

spreadbasis :
    Returns 'spread' basis for given Miller indices
"""

from core import dhkl, equivhkl, length, vol


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
    return


def nets(struct, indices):
    """Groups sites into dense nets normal to given hkl indices

    Parameters
    ----------


    Returns
    -------


    """
    # Using projection on dhkl, groups sites into dense nets.
    # Each net is representes as
    # [[sites numbers], width, eff. density].
    # Resulting list is sorted from max eff. density down.
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    def readcluster(Z, i, N):
        # For linkage matrix Z returns list of singletons
        # in Z[i] cluster and cluster width (sum of smallest
        # M-1 distances within cluster), where N, M - number of
        # singletons in total / in cluster.
        singletons = []
        width = Z[i][2]
        for j in Z[i][0:2]:
            if j < N:
                singletons.append(int(j))
            else:
                singletons += readcluster(Z, int(j) - N, N)[0]
                width += readcluster(Z, int(j) - N, N)[1]
        return [singletons, width]

    d = dhkl(struct.cell, indices)
    v = vol(struct.cell)[0]
    N = len(struct.p1().sites)
    if N == 1:
        return [[[0], 0, d/v]]

    projection = [(sum([j.fract[i]*indices[i] for i in range(3)]) % 1) * d
                  for j in struct.p1().sites]
    distances = pdist([[i] for i in projection],
                      lambda x, y: min(abs(x - y),
                                       abs(x - y + d),
                                       abs(x - y - d)))
    Z = linkage(distances, 'single')
    clusters = [readcluster(Z, j, N) for j in range(len(Z))]
    for i in clusters:
        i.append(len(i[0])*d/v * (1-i[1]/d*N/(len(i[0])-1)))
        # implementation of eff. density metrics
    for i in range(N):
        # accounting for single-site nets:
        clusters.append([[i], 0, d/v])
    clusters.sort(key=lambda x: x[2], reverse=True)
    result = []
    singletons = []
    for i in clusters:
        if len(set(i[0]) & set(singletons)) == 0:
            result.append(i)
            singletons += i[0]
    return result


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
