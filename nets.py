"""Module for analysis of 2D nets in crystal structures

Functions
---------
densenets : dict
    Clusterizes atomic nets for given hkl
    (monkeypatched as Structure method)
drawnet : matplotlib.pyplot.Figure
    Plots 2D Gabriel graph
    (monkeypatched as Structure method)
gabriel2d : core.Polyhedron
    Returns Gabriel neighbors
    (monkeypatched as Structure method)
med : float
    Returns mean effective density of nets
    (monkeypatched as Structure method)
searchnets : dict
    Returns mean eff. density for hkls in range
    (monkeypatched as Structure method)
spreadbasis :
    Returns 'spread' basis for given Miller indices
"""

from core import dhkl, equivhkl, length, maxdiag, orthonorm, Structure, vol


def densenets(self, indices, structures=True, fig=False):
    """Clusterizes atomic nets for given hkl

    Parameters
    ----------
    self : Structure
    indices : list or tuple
        [h, k, l]
    structures : bool
        whether create Structure for each cluster (default True)
    fig : bool
        whether include dendrogram in the output (default False)

    Returns
    -------
    dict
        {'cluster': [[...], [...], [...], ...],
         'width': [w1, w2, w3, ...],
         'density': [d1, d2, d3, ...],
         'z': [z1, z2, ...]
         'structure': [Structure1, Structure2, Structure3, ...],
         'fig': matplotlib.pyplot.Figure()}
        sorted by z in spread cell. When structures=True
        creates list of Structure instances corresponding
        to each cluster in spread cell. When fig=True
        creates a dendrogram.
    """

    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt

    def csd(x, y, d=1.):
        x = x % d
        y = y % d
        return min(abs(x - y), abs(x - y + d), abs(x - y - d))

    N = len(self.p1().sites)
    d = dhkl(self.cell, indices)
    V = vol(self.cell)[0]
    proj = [[(sum([j.fract[i]*indices[i] for i in range(3)]) % 1) * d]
            for j in self.p1().sites]
    distvec = pdist(proj,
                    metric=lambda x, y: csd(x, y, dhkl(self.cell, indices)))
    if len(self.p1().sites) == 1:
        Z = []
    else:
        Z = linkage(distvec, method='single')
    t = d / N
    clusters = []
    if len(Z) == 0:
        clusters = [[0]]
    else:
        leaves = fcluster(Z, t*0.999, criterion='distance')
        # 0.999 used to exclude clustering at threshold
        nclust = max(leaves)
        for i in range(1, nclust+1):
            cluster = []
            for j in range(len(leaves)):
                if leaves[j] == i:
                    cluster.append(j)
            clusters.append(cluster)

    zs = []  # z-coordinate of net in spread cell
    widths = []
    for i in clusters:
        z = []
        for j in i:
            if abs(proj[j][0] + d
                    - proj[i[0]][0]) < abs(proj[j][0] - proj[i[0]][0]):
                z.append(proj[j][0] + d)
            elif abs(proj[j][0] - d
                     - proj[i[0]][0]) < abs(proj[j][0] - proj[i[0]][0]):
                z.append(proj[j][0] - d)
            else:
                z.append(proj[j][0])
        zs.append(((sum(z) / len(z)) / d) % 1)

        matrix = squareform(distvec)
        width = 0
        for j in i:
            for k in i:
                if matrix[j][k] > width:
                    width = matrix[j][k]
        widths.append(width)

    densities = []
    for c, w in zip(clusters, widths):
        densities.append(
            len(c)*d/V * (1 - w/d*N / (len(c)-1)) if (len(c) > 1) else d/V)
        # implementation of effective density metrics
        # (for cluster of M sites equals M*d/V for zero width,
        # and 0 for width of d/N*(M-1))
    zs_sort, clusters_sort, densities_sort, widths_sort = zip(
        *sorted(list(zip(zs, clusters, densities, widths))))
    result = {'cluster': clusters_sort, 'density': densities_sort,
              'width': widths_sort, 'z': zs_sort}

    if structures:
        result['structure'] = []
        P = spreadbasis(indices, self.cell)
        for i in result['cluster']:
            result['structure'].append(
                self.p1().sublatt(i).transform(P))
    if fig:
        if len(Z) == 0:
            result['fig'] = None
        else:
            figure, axes = plt.subplots()
            t = dhkl(self.cell, indices) / len(self.p1().sites)
            dendrogram(Z, color_threshold=t*0.999,
                       labels=[i.label for i in self.p1().sites],
                       leaf_rotation=-90)
            axes.set_ylim(-t*0.05, t*2)
            axes.hlines(t, axes.viewLim.bounds[0],
                        axes.viewLim.bounds[0] + axes.viewLim.bounds[2],
                        linestyles='dashed')
            result['fig'] = figure
    return result


Structure.densenets = densenets


def drawnet(self, ax, cells=(2, 2)):
    """Plots 2D Gabriel graph

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        where to plot
    cells : tuple
        number of cells along a and b

    Returns
    -------
    matplotlib.pyplot.Figure
    """

    from itertools import product
    from numpy import array

    for ni, i in enumerate(self.p1().sites):
        # plain 'polyhedron' with direct neigbors:
        p = self.p1().gabriel2d(ni, flatten=True)
        for tr in product(range(cells[0]), range(cells[1])):
            i_coord = orthonorm(self.cell,
                                [i.fract[0]+tr[0], i.fract[1]+tr[1], 0, 1])
            ax.scatter([i_coord[0]], [i_coord[1]], c='C0', zorder=2)
            ax.text(i_coord[0], i_coord[1], i.label)
            for j in p.ligands:
                j_coord = orthonorm(self.cell,
                                    [j.fract[0]+tr[0], j.fract[1]+tr[1], 0, 1])
                ax.plot([i_coord[0], j_coord[0]], [i_coord[1], j_coord[1]],
                        alpha=0.75, zorder=1)

    uc = array([orthonorm(self.cell, i) for i in [[0, 0, 0, 1],
                                                  [0, 1, 0, 1],
                                                  [1, 1, 0, 1],
                                                  [1, 0, 0, 1],
                                                  [0, 0, 0, 1]]])
    ax.plot(uc[:, 0], uc[:, 1], 'k--', alpha=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    return


Structure.drawnet = drawnet


def gabriel2d(self, centr, ligands=None, flatten=False):
    """Returns Gabriel neighbors

    Parameters
    ----------
    centr : int
        number of central site in self.sites
    ligands : list
        numbers of ligands in self.sites
        (if None, all sites will be considered);
        default None
    flatten : bool
        whether apply z=0 for all sites
        (default False)

    Returns
    -------
    core.Polyhedron
    """

    if ligands is None:
        ligands = list(range(len(self.sites)))

    if flatten:
        from copy import deepcopy

        zeroz = deepcopy(self)
        for i in zeroz.sites:
            i.fract[2] = 0
        return zeroz.gabriel2d(centr, ligands)

    from numpy import array

    dmax = maxdiag(self.cell, plain=True)
    p = self.poly(centr=centr, ligands=ligands,
                  dmax=dmax, dmin=0.01, plain=True)
    blacklist = []
    for i in range(len(p.ligands)):
        for j in range(len(p.ligands)):
            if i != j:
                halfway = 0.5 * (array(p.central.fract)
                                 + array(p.ligands[i].fract))
                radius = 0.5 * length(self.cell, p.central.fract,
                                      p.ligands[i].fract, skipesd=True)[0]
                if length(self.cell, halfway, p.ligands[j].fract,
                          skipesd=True)[0] <= radius:
                    blacklist.append(i)
                    break
    for i in blacklist[::-1]:
        p.ligands.pop(i)
    return p


Structure.gabriel2d = gabriel2d


def med(self, indices, sublatt=None):
    """Returns mean effective density of nets

    Parameters
    ----------
    self : Structure
    indices : list or tuple
        [h, k ,l]
    sublatt : list
        [[a1, a2, ...], [b1, b2, ...], ...]
        sublattices to use (default None)

    Returns
    -------
    tuple
        (med, k_mix)
        k_mix (mixing coefficient) calculated in case of 2 sublattices
    """

    if sublatt is None:
        sublatt = [[i for i in range(len(self.sites))]]
    else:
        sublatt = [i for i in sublatt if i != []]
    full = []
    for i in sublatt:
        full += i
    full.sort()
    nets = self.sublatt(full).densenets(indices, structures=False)
    code = self.sublatt(full).p1_list()

    groups = []
    for s in sublatt:
        group = []
        for i in s:
            for j, k in enumerate(full):
                if i == k:
                    group += code[j]
        groups.append(group)

    partnets = []
    for g in groups:
        for c, d in zip(nets['cluster'], nets['density']):
            X = len(set(g) & set(c))
            partnets.append((X, len(c), d * X/len(c)))
    if len(sublatt) == 2:
        k_mix = (sum([(X/M)*((M-X)/M) / 0.25 * M for X, M, dpart
                      in partnets[:len(partnets)//2]])
                 / sum([M for X, M, dpart in partnets[:len(partnets)//2]]))
    else:
        k_mix = None
    return ((sum([X*dpart for X, M, dpart in partnets])
             / sum([X for X, M, dpart in partnets])), k_mix)


Structure.med = med


def searchnets(self, sublatt=None, hklmax=None):
    """Returns mean eff. density for hkls in range

    Parameters
    ----------
    self : Structure
    sublatt : list
        [[a1, a2, ...], [b1, b2, ...], ...]
        sublattices to use (default None)
    hklmax : list
        max abs of used h, k, l (default None
        interpreted as [5, 5, 5])

    Returns
    -------
    DataFrame
        {'hkl': [[h1, k1, l1], [h2, k2, l2], ...],
         'mean': [dmean1, dmean2, ...],
         'equiv': [[[h, k, l], [...], ...], [[...], [...], ...], ...]}
        (equiv corresponds to equivalent hkls)
    """

    if hklmax is None:
        hklmax = [5, 5, 5]

    from math import gcd
    from pandas import DataFrame
    import time

    print(f'searchnets started with hklmax={hklmax}')
    starttime = time.time()

    hkls = []
    equivs = []
    for h in range(hklmax[0], -hklmax[0]-1, -1):
        for k in range(hklmax[1], -hklmax[1]-1, -1):
            for l in range(hklmax[2], -hklmax[2]-1, -1):
                indices = (h, k, l)
                # tuple used for further handling in pandas DataFrame
                if gcd(*indices) != 1:
                    continue
                equiv = equivhkl(self.symops, indices, laue=True)
                if len([x for x in equiv if tuple(x) in hkls]) != 0:
                    continue
                else:
                    hkls.append(indices)
                    equivs.append(equiv)
    print(f'hkl list prepared ({len(hkls)} indices)')

    means = []
    k_mixs = []
    counter = 1  # used only in messages
    for indices in hkls:
        m, k = self.med(indices, sublatt)
        means.append(m)
        k_mixs.append(k)
        elapsed = time.time() - starttime
        remaining = int(elapsed/counter * len(hkls) - elapsed)
        print(f'\x1b[2K{indices} completed ({counter} of {len(hkls)}): '
              f'{elapsed:.1f} s elapsed, ~{remaining} s remaining\r', end='')
        counter += 1
    print(f'\nsearchnets completed in {time.time()-starttime:.1f} s\n')
    return DataFrame({'hkl': hkls, 'mean': means, 'k_mix': k_mixs,
                      'equiv': equivs})


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
