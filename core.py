"""Module for analysis of crystal structure geometry

Classes
-------
Polyhedron
    Defines coordination polyhedron
Site
    Defines atomic site in crystal structure
Structure
    Defines crystal structure

Functions
---------
angle
    Returns angle and its esd
clearkeys
    Removes empty keys from CIF-based dict
cumdiff
    Calculates difference between distance distributions
dhkl
    Returns interplanar distance
equivhkl
    Returns list of equivalent Miller indices
lattrange
    Returns lattice range covering sphere of given radius
length
    Returns length and its esd
matrixform
    Returns matrix form of symmetry operations
newbasis
    Returns transformed basis [a, b, c, al, be, ga] and esds
orthonorm
    Returns orthonormal coordinates of point
parsecif
    Returns CIF content
qimport
    Quick wrapper for parsecif
readesd
    Returns value and its esd from string with parentheses
readformula
    Interprets _chemical_formula_sum value
readstruct
    Returns Structure object from CIF-based dict
stringform
    Returns string form of symmetry operation
vol
    Returns unit cell volume
writesd
    Returns value and its eds as string with parentheses
"""

# CIF keys used in Structure constructor
whitelist_structure = ["_cell_length_a",
                       "_cell_length_b",
                       "_cell_length_c",
                       "_cell_angle_alpha",
                       "_cell_angle_beta",
                       "_cell_angle_gamma",
                       "_space_group_symop_operation_xyz",
                       "_symmetry_equiv_pos_as_xyz",
                       "_atom_site_label",
                       "_atom_site_type_symbol",
                       "_atom_site_fract_x",
                       "_atom_site_fract_y",
                       "_atom_site_fract_z",
                       "_atom_site_occupancy",
                       '_atom_site_b_iso_or_equiv',
                       '_atom_site_u_iso_or_equiv']


class Polyhedron:
    """Defines coordination polyhedron

    Attributes
    ----------
    central : Site
        central site of the polyhedron
    ligands : list
        list of ligands (Site instances)
    cell : list
        [a, b, c, alpha, beta, gamma]
    cell_esd : list
        [a_esd, b_esd, c_esd, alpha_esd, beta_esd, gamma_esd]

    Methods
    -------
    bondweights : dict
        Returns bond weights in CHARDI approach
    econ : tuple
        Returns effective coordination number and its esd
    hidden : int
        Returns number of hidden ligands
    listangl : dict
        Returns angles and their esds in polyhedron
    listdist : dict
        Returns distances and their esds in polyhedron
    listdist_corr : dict
        Returns U-corrected distances and their esds in polyhedron
    meandist : tuple
        Returns mean distance and its esd in polyhedron
    polyvol : tuple
        Returns polyhedron volume and its esd
    polyvol_corr : tuple
        Returns U-corrected polyhedron volume and its esd
    wmd : float
        Returns weighted mean distance (CHARDI)
    """

    def __init__(self, central, ligands, cell,
                 cell_esd=None):
        """
        Parameters
        ----------
        central : Site
            central site of the polyhedron
        ligands : list
            list of ligands (Site instances)
        cell : list
            [a, b, c, alpha, beta, gamma]
        cell_esd : list
            [a_esd, b_esd, c_esd, alpha_esd, beta_esd, gamma_esd]
            (default None interpreted as [0, 0, 0, 0, 0, 0])
        """

        self.central = central
        self.ligands = ligands
        self.cell = cell
        if cell_esd is None:
            self.cell_esd = [0, 0, 0, 0, 0, 0]
        else:
            self.cell_esd = cell_esd

    def __repr__(self):
        lig = ''
        for i, j, k in zip(self.ligands,
                           self.listdist()['value'],
                           self.listdist()['esd']):
            lig += f'{i.label} ({i.symbol}) {writesd(j, k, ".3f")}\n'
        return (f'{self.central.label} ({self.central.symbol})\n{lig[:-1]}')

    def bondweights(self):
        """Returns bond weights in CHARDI approach

        Convergence condition: max bond weight change < 0.01.

        Returns
        -------
        dict
            {"name": [], "value": [], "esd": []}
        """

        from math import exp

        result = {"name": [], "value": [], "esd": []}
        if len(self.ligands) == 0:
            return result
        ld = self.listdist()
        result['name'] = ld['name']
        dist = ld['value']
        dist_esd = ld['esd']
        d_ave = min(dist)
        weight_old = [1.0 for i in dist]
        for _ in range(100):
            weight = [exp(1 - (i / d_ave)**6) for i in dist]
            weight_esd = [i * j * 6*k**5 / d_ave**6
                          for i, j, k in zip(dist_esd, weight, dist)]
            # esd of d_ave is ignored
            delta = [abs(i - j) for i, j in zip(weight, weight_old)]
            if max(delta) < 0.001:
                break
            d_ave = sum([i*j for i, j in zip(dist, weight)]) / sum(weight)
            weight_old = weight
        result['value'] = weight
        result['esd'] = weight_esd
        return result

    def econ(self):
        """Returns effective coordination number

        Returns
        -------
        tuple
            (econ, esd)
        """

        bw = self.bondweights()
        esd = sum([i**2 for i in bw['esd']])**0.5
        return (sum(bw['value']), esd)

    def hidden(self):
        """Returns number of hidden ligands

        Ligands inside convex hull are considered as hidden.

        Returns
        -------
        int
            number of hidden ligands
        """

        from scipy.spatial import ConvexHull, QhullError

        if len(self.ligands) <= 4:
            return 0
        else:
            vert_ort = [orthonorm(self.cell, i.fract) for i in self.ligands]
            try:
                return len(self.ligands) - len(ConvexHull(vert_ort).vertices)
            except QhullError:
                return 0

    def listangl(self):
        """Returns angles and their esds in polyhedron

        Returns
        -------
        dict
            {"name": [], "value": [], "esd": []}
        """

        result = {"name": [], "value": [], "esd": []}
        # sorting by label to avoid A-B-C and C-B-A dichotomy
        lig_sorted = sorted(self.ligands, key=lambda x: x.label)
        for n, i in enumerate(lig_sorted):
            for m, j in enumerate(lig_sorted):
                if m > n:
                    result["name"].append(
                        f"{i.label}-{self.central.label}-{j.label}")
                    a = angle(self.cell, self.central.fract, i.fract, j.fract,
                              self.cell_esd, self.central.fract_esd,
                              i.fract_esd, j.fract_esd)
                    result["value"].append(a[0])
                    result["esd"].append(a[1])
        return result

    def listdist(self, skipesd=False):
        """Returns distances and their esds in polyhedron

        Parameters
        ----------
        skipesd : bool
            if True, skips esd evaluation (default False)

        Returns
        -------
        dict
            {"name": [], "value": [], "esd": []}
        """

        result = {"name": [], "value": [], "esd": []}
        for i in self.ligands:
            result["name"].append(f"{self.central.label}-{i.label}")
            d = length(self.cell, self.central.fract, i.fract,
                       self.cell_esd, self.central.fract_esd, i.fract_esd,
                       skipesd=skipesd)
            result["value"].append(d[0])
            result["esd"].append(d[1])
        return result

    def listdist_corr(self):
        """Returns U-corrected distances and their esds in polyhedron

        'Simple rigid bond' approximation (Downs, 2000) is used

        Returns
        -------
        dict
            {"name": [], "value": [], "esd": []}
        """

        result = self.listdist()
        Ucen = self.central.u
        Ucen_esd = self.central.u_esd
        for i, lig in enumerate(self.ligands):
            Ulig = lig.u
            Ulig_esd = lig.u_esd
            dobs = result['value'][i]
            dobs_esd = result['esd'][i]
            dcorr = (dobs**2 + Ulig - Ucen)**0.5
            dcorr_esd = (dobs_esd**2 * 4 * dobs**2
                         + Ulig_esd**2 + Ucen_esd**2)**0.5 / 2 / dcorr
            result["value"][i] = dcorr
            result["esd"][i] = dcorr_esd
        return result

    def meandist(self):
        """Returns mean distance and its esd in polyhedron

        Returns
        -------
        tuple
            (mean distance, esd)
        """

        from statistics import mean, stdev

        return (mean(self.listdist()['value']),
                stdev(self.listdist()['value']))

    def polyvol(self):
        """Returns polyhedron volume and its esd

        Relative volume esd is calculated as three times
        relative radius esd, the latter being mean relative esd
        of distances from polyhedron center to ligands.
        Polyhedron center comes from averaging of ligand
        coordinates and not from central atom!
        Error in center determination is not considered
        in esd calculation, so the resulting esd is OVERestimated
        due to correlation between ligand and center coordinates.

        Returns
        -------
        tuple
            (volume, esd)
        """

        from scipy.spatial import ConvexHull, QhullError

        N = len(self.ligands)
        if N <= 3:
            return (0, 0)
        vert_ort = [orthonorm(self.cell, i.fract) for i in self.ligands]
        outer_lig = []
        try:
            for i in ConvexHull(vert_ort).vertices:
                outer_lig.append(self.ligands[i])
        except QhullError:
            return (0, 0)
        N = len(outer_lig)
        center = [sum([outer_lig[j].fract[i] for j in range(N)]) / N
                  for i in range(3)]
        distances = [length(self.cell, center, i.fract,
                            self.cell_esd, [0, 0, 0, 0], i.fract_esd)
                     for i in outer_lig]
        rel_dist_esd = sum([i[1] / i[0] for i in distances]) / N
        try:
            vol = ConvexHull(vert_ort).volume
        except QhullError:
            return (0, 0)
        return (vol, vol * 3*rel_dist_esd)

    def polyvol_corr(self):
        """Returns U-corrected polyhedron volume and its esd

        V_corr = V * (1 + 3*(r_corr - r)/r), where V - uncorrected value,
        (r_corr - r)/r - relative increase of mean distance in polyhedron
        due to U correction; esd is not modified.

        Returns
        -------
        tuple
            (volume, esd)
        """

        V, esd = self.polyvol()
        if V == 0:
            return (0, 0)
        r = sum(self.listdist()['value'])
        r_corr = sum(self.listdist_corr()['value'])
        return (V * (1 + 3*(r_corr - r)/r), esd)

    def wmd(self):
        """Weighted mean distance (CHARDI)

        Returns
        -------
        tuple
            value, esd
        """

        from numpy import exp
        from pandas import DataFrame

        ld = DataFrame(self.listdist())
        if len(ld['value']) == 0:
            return None, None
        d0 = min(ld['value'])
        for i in range(100):
            ld['weight'] = exp(1 - (ld['value']/d0)**6)
            d0_new = (ld['value']*ld['weight']).sum() / ld['weight'].sum()
            if abs(d0 - d0_new) < 0.001:
                # weights' esds are ignored:
                return d0_new, (((ld['esd']*ld['weight'])**2).sum())**0.5
            d0 = d0_new
        print('Error in wmd: convergence not reached!')
        return None, None


class Site:
    """Defines atomic site in crystal structure

    Attributes
    ----------
    fract : list
        fractional coordinates in augmented form ([x, y, z, 1])
    fract_esd : list
        esd of fractional coordinates in augmented form ([x, y, z, 0])
    label : str
        site label ('_atom_site_label' CIF key)
    symbol : str
        site symbol ('_atom_site_type_symbol' CIF key)
    occ : float
        site occupancy
    occ_esd : float
        occupancy esd
    u : float
        Uiso / Uequiv
    u_esd : float
        esd of Uso / Uequiv
    """

    def __init__(self, fract, fract_esd=None,
                 label="H1", symbol="H", occ=1.0, occ_esd=0.0,
                 u=0.0, u_esd=0.0):
        """
        Parameters
        ----------
        fract : list
            fractional coordinates in augmented form ([x, y, z, 1])
        fract_esd : list
            esd of fractional coordinates in augmented form ([x, y, z, 0]),
            default None interpreted as [0, 0, 0, 0]
        label : str
            site label ('_atom_site_label' CIF key), default 'H1'
        symbol : str
            site symbol ('_atom_site_type_symbol' CIF key), default 'H'
        occ : float
            site occupancy (default 1.0)
        occ_esd : float
            occupance esd (default 0.0)
        u : float
            Uiso / Uequiv
        u_esd : float
            esd of Uso / Uequiv
        """

        self.fract = fract
        if fract_esd is None:
            self.fract_esd = [0, 0, 0, 0]
        else:
            self.fract_esd = fract_esd
        self.label = label
        self.symbol = symbol
        self.occ = occ
        self.occ_esd = occ_esd
        self.u = u
        self.u_esd = u_esd

    def __repr__(self):
        x, y, z = [writesd(i, j) for i, j in zip(self.fract[:3],
                                                 self.fract_esd)]
        return (f'{self.label} ({self.symbol}) at {x}, {y}, {z} '
                f'(occ: {writesd(self.occ, self.occ_esd)}, '
                f'Uiso/eq: {writesd(self.u, self.u_esd)})')


class Structure:
    """Defines crystal structure

    Attributes
    ----------
    cell : list
        [a, b, c, alpha, beta, gamma]
    cell_esd : list
        [a_esd, b_esd, c_esd, alpha_esd, beta_esd, gamma_esd]
    sites : list
        list of Site instances
    symops : list
        list of symmetry operations (4*4 augmented matrices)

    Methods
    -------
    cif : str
        Outputs content of Structure instance in CIF format
    filter : list
        Returns numbers of self.sites present in values list
    formula : dict
        Returns formula unit
    p1 : Structure
        Returns geometrically equivalent structure with P1 space group
    p1_list : list
        Returns numbers of p1() sites equivalent to the original ones
    pairs : list
        Returns distances in structure
    poly : Polyhedron
        Returns Polyhedron instance with given central site and ligands
    resymbol : None
        Standartise chemical element symbols
    skipdupl : Structure
        Removes duplicate sites (e.g. disordered ones)
    sublatt : Structure
        Returns sublattice of structure
    transform : Structure
        Changes basis
    """

    def __init__(self,
                 cell=None,
                 cell_esd=None,
                 sites=None,
                 symops=None):
        """
        Parameters
        ----------
        cell : list
            [a, b, c, alpha, beta, gamma]
            (default None interpreted as cube of unit size)
        cell_esd : list
            [a_esd, b_esd, c_esd, alpha_esd, beta_esd, gamma_esd]
            (default None interpretes as [0, 0, 0, 0, 0, 0])
        sites : list
            list of Site instances (default None interpreted as [])
        symops : list
            list of symmetry operations (4*4 augmented matrices)
            (default None interpreted as identity)
        """

        if cell is None:
            self.cell = [1, 1, 1, 90, 90, 90]
        else:
            self.cell = cell
        if cell_esd is None:
            self.cell_esd = [0, 0, 0, 0, 0, 0]
        else:
            self.cell_esd = cell_esd
        if sites is None:
            self.sites = []
        else:
            self.sites = sites
        if symops is None:
            self.symops = [[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]]
        else:
            self.symops = symops

    def __repr__(self):
        return self.cif()

    def cif(self, datablock='I'):
        """Prints Structure contents in CIF syntax

        Parameters
        ----------
        datablock : str
            name used in _data key (default I)

        Returns
        -------
        str
        """

        symops = ''
        for i, s in enumerate(self.symops):
            symops += f'{i+1} {stringform(s)}\n'
        sites = ''
        for s in self.sites:
            x, y, z = [writesd(i, j, '.6f') for i, j
                       in zip(s.fract[:3], s.fract_esd)]
            sites += (f'{s.label} {s.symbol} {x} {y} {z} '
                      f'{writesd(s.occ, s.occ_esd)} {writesd(s.u, s.u_esd)}\n')
        result = (
            f"data_{datablock}\n"
            f"_cell_length_a {writesd(self.cell[0], self.cell_esd[0])}\n"
            f"_cell_length_b {writesd(self.cell[1], self.cell_esd[1])}\n"
            f"_cell_length_c {writesd(self.cell[2], self.cell_esd[2])}\n"
            f"_cell_angle_alpha {writesd(self.cell[3], self.cell_esd[3])}\n"
            f"_cell_angle_beta {writesd(self.cell[4], self.cell_esd[4])}\n"
            f"_cell_angle_gamma {writesd(self.cell[5], self.cell_esd[5])}\n"
            f"loop_\n"
            f"_space_group_symop_id\n"
            f"_space_group_symop_operation_xyz\n"
            f'{symops}'
            f"loop_\n"
            f"_atom_site_label\n"
            f"_atom_site_type_symbol\n"
            f"_atom_site_fract_x\n"
            f"_atom_site_fract_y\n"
            f"_atom_site_fract_z\n"
            f"_atom_site_occupancy\n"
            f"_atom_site_u_iso_or_equiv\n"
            f'{sites[:-1]}'
        )
        return result

    def filter(self, key, values, inverse=False):
        """Returns numbers of self.sites present in values list

        Parameters
        ----------
        key : str
            'label', 'symbol' or 'number'
        values : list
            list of key values
        inverse : bool
            inverse selection

        Returns
        -------
        list
            numbers of required self.sites
        """

        if key != 'number':
            values = [i.lower() for i in values]
        result = []
        for i in enumerate(self.sites):
            if not inverse:
                if (key == 'label') and (i[1].label.lower() in values):
                    result.append(i[0])
                elif (key == 'symbol') and (i[1].symbol.lower() in values):
                    result.append(i[0])
                elif (key == 'number') and (i[0] in values):
                    result.append(i[0])
            else:
                if (key == 'label') and not (i[1].label.lower() in values):
                    result.append(i[0])
                elif (key == 'symbol') and not (i[1].symbol.lower() in values):
                    result.append(i[0])
                elif (key == 'number') and not (i[0] in values):
                    result.append(i[0])
        return result

    def formula(self, z=1):
        """Returns formula unit

        Parameters
        ----------
        z : float
            number of formula units in unit cell

        Returns
        -------
        dict
            {symbol: content, ...}

        """

        result = {}
        for i in set([j.symbol for j in self.sites]):
            result[i] = sum([k.occ if k.symbol == i else 0
                             for k in self.p1().sites]) / z
        return result

    def p1(self, delta=0.01, symkeys=False):
        """Returns geometrically equivalent structure with P1 space group

        Parameters
        ----------
        delta : float
            min difference between symmetrically equivalent sites
            in angstroms (default 0.01)
        symkeys : bool
            whether symmetry key (symop #) will be added to
            equivalent site labels (default False)

        Returns
        -------
        Structure
        """

        from copy import deepcopy
        from numpy import array, diag, matmul

        sites_p1 = []
        for i in self.sites:
            equiv = []
            for j in range(len(self.symops)):
                symkey = ""
                if symkeys:
                    symkey = "_"+str(j+1)
                newsite = deepcopy(i)
                newsite.fract = [sum([row[k]*newsite.fract[k]
                                      for k in range(4)]) % 1
                                 # reducing to single cell
                                 for row in self.symops[j]]
                newsite.fract_esd = [(row**2).sum()**0.5
                                     for row in
                                     matmul(array(self.symops[j]),
                                            diag(newsite.fract_esd))]
                newsite.label += symkey
                equiv.append(newsite)
            for j in range(len(equiv) - 1, 0, -1):
                kill = False
                for k in range(j - 1, -1, -1):
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                x, y, z = equiv[k].fract[:3]
                                if length(self.cell, equiv[j].fract,
                                          [x+dx, y+dy, z+dz, 1])[0] < delta:
                                    kill = True
                                    break
                            if kill:
                                break
                        if kill:
                            break
                    if kill:
                        equiv.pop(j)  # removing duplicate equivalents
                        break
            sites_p1 += equiv
        result = deepcopy(self)
        result.symops = [[[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]]
        result.sites = sites_p1
        return result

    def p1_list(self, delta=0.01):
        """Returns numbers of p1() sites equivalent to the original ones

        Parameters
        ----------
        delta : float
            min difference between symmetrically equivalent sites
            in angstroms (default 0.01)

        Returns
        -------
        list
            [[p1, p2, ...], [pn, pn+1, ...], ...]
            with list for each of self.sites
        """

        result = []
        for i in self.sites:
            equiv = []
            for j in range(len(self.symops)):
                fract = [sum([row[k]*i.fract[k]
                              for k in range(4)]) % 1
                         # reducing to single cell
                         for row in self.symops[j]]
                equiv.append(fract)
            counter = len(equiv)
            for j in range(len(equiv) - 1, 0, -1):
                kill = False
                for k in range(j - 1, -1, -1):
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                x, y, z = equiv[k][:3]
                                if length(self.cell, equiv[j],
                                          [x+dx, y+dy, z+dz, 1],
                                          skipesd=True)[0] < delta:
                                    kill = True
                                    break
                            if kill:
                                break
                        if kill:
                            break
                    if kill:
                        counter -= 1
                        break
            if len(result) == 0:
                result.append(list(range(counter)))
            else:
                result.append(list(range(result[-1][-1]+1,
                                         result[-1][-1]+1 + counter)))
        return result

    def pairs2(self, dmax, A, B):
        """Returns distances in structure

        Parameters
        ----------
        dmax : float
            max distance
        A : list
            sublattice of central atoms
        B : list
            sublattice of ligands

        Returns
        -------
        pd.Series
            sorted list of distances
        """

        from pandas import Series

        # multiplicities of sites:
        mult = [len(i) for i in self.p1_list()]
        distances = []
        for i in A:
            distances += self.poly(
                i, B, dmax, dmin=0.01, distonly=True
            )['value']*mult[i]

        return Series(distances).sort_values()

    def pairs_full(self, dmax, SL=None, prec=3):
        """Returns distances in structure

        Parameters
        ----------
        dmax : float
            max distance
        SL : list
            list of sublattices
            (default None)
        prec : int
            decimals in normalized distances

        Returns
        -------
        pd.Series
            index: sorted (distances, types)
            value: normalized number of distances
            (keys of distance types correspond to
            sequence of cells in triangular matrix
            of sublattice pairs)
        """

        from numpy import pi
        from pandas import DataFrame

        if SL is None:
            SL = [
                list(range(len(self.sites)))
            ]

        # multiplicity of all sites:
        mult = [len(i) for i in self.p1_list()]
        # multiplicities of sublattices:
        mult_SL = [sum([mult[i] for i in j]) for j in SL]
        result = {}
        n = 0
        for i in range(len(SL)):
            for j in range(i, len(SL)):
                if (mult_SL[i] != 0) and (mult_SL[i] != 0):
                    # distance normalization factor:
                    f = (
                        (mult_SL[i] + mult_SL[j])
                        / (2 if (j == i) else 1)
                        / vol(self.cell)[0]
                    ) ** (1/3)
                    # normalized distances:
                    nd = (
                        self.pairs2(dmax / f, SL[i], SL[j]) * f
                    ).round(prec).value_counts()
                    result[n] = (nd * 2
                                 / (mult_SL[i] + mult_SL[j])
                                 / (4 * pi * nd.index**2)
                                 )
                    # weights of sublattice pairs
                    # are encoded under zero index:
                    result[n].loc[0] = ((mult_SL[i] * mult_SL[j])
                                        * (1 if (j == i) else 2)
                                        / sum(mult_SL)**2)
                    result[n].sort_index(inplace=True)
                n += 1
        return DataFrame(result).stack().sort_index()

    def poly(self, centr, ligands, dmax, dmin=0.0,
             nmax=None, suffixes=False, plain=False,
             distonly=False):
        """Returns Polyhedron

        Parameters
        ----------
        centr : int
            number of central site in self.sites
        ligands : list
            numbers of ligands in self.sites
        dmax : float
            upper limit of bond length in Angstroms
        dmin : float
            lower limit of bond length in Angstroms (default 0.0)
        nmax : int
            limits max number of ligands to nmax closest ones
            unless None (default)
        suffixes : bool
            if True, suffixes with symmetry operations will be
            attached to ligands' labels (default False)
        plain : bool
            if True, Nc will be set to zero for plain nets
            analysis (default False)
        distonly : bool
            if True, returns only list of distances
            in Polyhedron.listdist() format (default False)

        Returns
        -------
        Polyhedron
        """

        from copy import deepcopy

        centr_site = deepcopy(self.sites[centr])
        centr_site.fract = [i % 1 for i in centr_site.fract]
        # reduction to the first cell
        liglist = []
        Na, Nb, Nc = lattrange(self.cell, dmax)
        if plain:
            Nc = 0
        for s in self.sublatt(ligands).p1(symkeys=suffixes).sites:
            for i in range(-Na, Na+1):
                for j in range(-Nb, Nb+1):
                    for k in range(-Nc, Nc+1):
                        dist = length(self.cell,
                                      centr_site.fract,
                                      [s.fract[0]+i,
                                       s.fract[1]+j,
                                       s.fract[2]+k,
                                       1],
                                      skipesd=True)
                        if dmin <= dist[0] <= dmax:
                            if suffixes:
                                suffix = f"_{i}{j}{k}"
                            else:
                                suffix = ""
                            if distonly:
                                newsite = None
                            else:
                                newsite = deepcopy(s)
                                newsite.fract = [newsite.fract[0]+i,
                                                 newsite.fract[1]+j,
                                                 newsite.fract[2]+k,
                                                 1]
                                newsite.label += suffix
                            liglist.append(
                                [newsite, dist,
                                 f"{centr_site.label}-{s.label+suffix}"])
        if distonly:
            out = {'name': [], 'value': []}
            for i in liglist:
                out['name'].append(i[2])
                out['value'].append(i[1][0])
            return out
        if liglist == []:
            return Polyhedron(centr_site, [], self.cell, self.cell_esd)
        liglist.sort(key=lambda x: x[1][0])
        if (nmax is None) or (nmax > len(liglist)):
            nmax = len(liglist)
        return Polyhedron(centr_site, [i[0] for i in liglist[:nmax]],
                          self.cell, self.cell_esd)

    def resymbol(self):
        """Standartise chemical element symbols"""

        elements = [
            'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bh',
            'Bi', 'Bk', 'Br', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co',
            'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'Fe', 'Fl',
            'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'In',
            'Ir', 'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg', 'Mn',
            'Mo', 'Mt', 'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni', 'No', 'Np', 'Og',
            'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb',
            'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru', 'Sb', 'Sc', 'Se', 'Sg', 'Si',
            'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm',
            'Ts', 'Xe', 'Yb', 'Zn', 'Zr',
            'B', 'C', 'F', 'H', 'I', 'K', 'N',
            'O', 'P', 'S', 'U', 'V', 'W', 'Y'
        ]
        for s in self.sites:
            for e in elements:
                if s.symbol.lower().startswith(e.lower()):
                    s.symbol = e
                    break

    def skipdupl(self, dmin=0.1):
        """Removes duplicate sites (e.g. disordered ones)

        Returns
        -------
        Structure
            Structure.p1() instance with removed duplicates
        """

        from numpy import array

        if self.symops == [[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]]:
            struct = self
        else:
            struct = self.p1()
        N = len(struct.sites)
        blacklist = []
        for i in range(N-1):
            for j in range(i+1, N):
                # check at cell boundaries:
                for a in [-1, 0, 1]:
                    for b in [-1, 0, 1]:
                        for c in [-1, 0, 1]:
                            if length(struct.cell, struct.sites[i].fract,
                                      (array(struct.sites[j].fract)
                                       + array([a, b, c, 0])),
                                      skipesd=True)[0] < dmin:
                                blacklist.append(j)
        return struct.sublatt(
            set(range(N)).difference(set(blacklist))
        )

    def sublatt(self, filter):
        """Returns sublattice of structure

        Parameters
        ----------
        filter : list
            numbers of sites to keep

        Returns
        -------
        Structure
        """

        from copy import deepcopy

        sites_sub = []
        for i in range(len(self.sites)):
            if i in filter:
                sites_sub.append(deepcopy(self.sites[i]))
        result = deepcopy(self)
        result.sites = sites_sub
        return result

    def transform(self, P):
        """Changes basis

        Parameters
        P : list
            4*4 transformation P matrix as defined in Muller (2013)
            (note column by column order)

        Returns
        -------
        Structure
        """

        from copy import deepcopy
        from numpy import array, diag, matmul
        from numpy.linalg import inv

        P = array(P)
        Pinv = inv(array(P))
        new_cell, new_cell_esd = newbasis(self.cell, P, self.cell_esd)
        new_symops = []
        for i in self.symops:
            new_symops.append(matmul(Pinv, matmul(i, P)))
        new_sites = []
        for i in self.sites:
            s = deepcopy(i)
            s.fract = [sum([row[k]*s.fract[k]
                            for k in range(4)]) % 1
                       # reducing to single cell
                       for row in Pinv]
            s.fract_esd = [(row**2).sum()**0.5
                           for row in matmul(Pinv, diag(s.fract_esd))]
            # TODO: check!
            new_sites.append(s)
        return Structure(new_cell, new_cell_esd, new_sites, new_symops)


def angle(cell, u, v, w,
          cell_esd=None,
          u_esd=None,
          v_esd=None,
          w_esd=None):
    """Returns angle between v-u and w-u vectors in given basis

    Parameters
    ----------
    cell : list
        [a, b, c, alpha, beta, gamma]
    u : list
        fractional coordinates of u
    v : list
        fractional coordinates of v
    w : list
        fractional coordinates of w
    cell_esd : list
        esds of cell (default None interpreted
        as [0, 0, 0, 0, 0, 0])
    u_esd : list
        esds of u (default None interpreted
        as [0, 0, 0, 0])
    v_esd : list
        esds of v (default None interpreted
        as [0, 0, 0, 0])
    w_esd : list
        esds of w (default None interpreted
        as [0, 0, 0, 0])

    Returns
    -------
    tuple
        (angle, esd)
    """

    if cell_esd is None:
        cell_esd = [0, 0, 0, 0, 0, 0]
    if u_esd is None:
        u_esd = [0, 0, 0, 0]
    if v_esd is None:
        v_esd = [0, 0, 0, 0]
    if w_esd is None:
        w_esd = [0, 0, 0, 0]

    from math import acos, pi, sqrt

    a, a_esd = length(cell, v, w, cell_esd, v_esd, w_esd)
    b, b_esd = length(cell, u, v, cell_esd, u_esd, v_esd)
    c, c_esd = length(cell, u, w, cell_esd, u_esd, w_esd)
    if (b == 0) or (c == 0):
        return (None, None)
    # check for rounding errors
    if (b**2 + c**2 - a**2)/2/b/c < -1.0:
        al = 180
    elif (b**2 + c**2 - a**2)/2/b/c > 1.0:
        al = 0
    else:
        al = acos((b**2 + c**2 - a**2)/2/b/c)*180/pi
    # from Giacovazzo (1992), p. 123
    al_esd = 180/pi*sqrt(a_esd**2 * a**2 / b**2 / c**2
                         + b_esd**2 / b**2
                         + c_esd**2 / c**2)
    return (al, al_esd)


def clearkeys(data, loops=None):
    """Removes empty keys from CIF-based dict

    Removes keys with '?', ['?', '?', ...], '.', and ['.', '.', ...] values
    ignoring leading/traling whitespaces and updates loops list accordingly

    Parameters
    ----------
    data : dict
        CIF-based dict {keys: values}
    loops : list
        [[loop1 keys], [loop2 keys], ...]

    Returns
    -------
    tuple
        (data_cleared, loops_cleared)
    """

    data_cleared = {}
    for i in data:
        if type(data[i]) is str:  # ignoring loops (lists)
            if (data[i].strip() == '?') or (data[i].strip() == '.'):
                continue
        elif (([j.strip() for j in data[i]] == ['?']*len(data[i]))
              or ([j.strip() for j in data[i]] == ['.']*len(data[i]))):
            continue
        data_cleared[i] = data[i][:]  # supports both str and lists
    if loops is None:
        loops_cleared = None
    else:
        loops_cleared = [i[:] for i in loops]
        for i, j in enumerate(loops):
            for k in j:
                if k not in data_cleared:
                    loops_cleared[i].remove(k)
        for i in loops_cleared[:]:
            if i == []:
                loops_cleared.remove(i)
    return (data_cleared, loops_cleared)


def cumdiff2(S1, S2, raw=False):
    """Calculates r.m.s. difference between cumulative distributions

    Assumes common distance range of distributions!

    Parameters
    ----------
    S1 : pandas.Series
        distance statistics
        (Structure.pairs2() output)
    S2 : pandas.Series
        distance statistics
        (Structure.pairs2() output)
    raw : bool
        if True, returns raw data
        (default False)

    Returns
    -------
    float
        r.m.s. difference between cumulative distributions
    """

    from pandas import DataFrame

    X = S1.dropna().sort_index().cumsum()
    Y = S2.dropna().sort_index().cumsum()
    X.name = 'X'
    Y.name = 'Y'
    df = DataFrame([X, Y]).transpose().sort_index()
    if len(df.index) == 0:
        return 0.

    df['d'] = df.index
    if len(df.index) > 1:
        df['d_next'] = df['d'].shift(-1)
    else:
        df['d_next'] = df['d']
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    # weighting by 1/x**2:
    df['w'] = (df['d_next'] - df['d']) / df['d_next'] / df['d']

    if raw:
        return df

    return (
        ((df['X'] - df['Y'])**2 * df['w']).sum()
        / (1/df.index.min() - 1/df.index.max())
    )**0.5


def cumdiff_full(S1, S2, M=None):
    """Calculates weighted r.m.s. difference for ND-PDFs

    Assumes common distance range of distributions!

    Parameters
    ----------
    S1 : pandas.Series
        distance statistics
        (Structure.pairs_full() output)
    S2 : pandas.Series
        distance statistics
        (Structure.pairs_full() output)
    M : dict
        correspondence between sublattices

    Returns
    -------
    dict
        r.m.s. difference for sublattice pairs
        and weighted average
    """

    # weights of sublattice pairs:
    w1 = S1.loc[0.]
    w2 = S2.loc[0.]

    if M is None:
        M = {i: i for i in range(len(w1))}
    # renormalization:
    w1 = w1[w1.index.isin(M)].values
    w1 /= w1.sum()
    w2 = w2[w2.index.isin(M.values())].values
    w2 /= w2.sum()
    weights = (w1 + w2) / 2
    diffs = []
    for i in M:
        diffs.append(
            cumdiff2(S1.unstack()[i].iloc[1:],
                     S2.unstack()[M[i]].iloc[1:])
        )

    return {i: diffs[i] for i in M} | {'w': sum(
        [diffs[i] * weights[i] for i in M]
    )}


def dhkl(cell, indices):
    """Returns interplanar distance

    After Vainshteyn (1979)

    Parameters
    ----------
    cell : list
        [a, b, c, alpha, beta, gamma]
    indices : list
        [h, k, l]

    Returns
    -------
    float
        d-spacing
    """

    from math import cos, sin, sqrt, pi

    a, b, c = cell[:3]
    al, be, ga = [i/180*pi for i in cell[3:]]
    h, k, l = indices
    if h == 0 and k == 0 and l == 0:
        return None
    return sqrt(
        (1 - cos(al)**2 - cos(be)**2 - cos(ga)**2
         + 2*cos(al)*cos(be)*cos(ga)) / (
             h**2 / a**2 * sin(al)**2
             + k**2 / b**2 * sin(be)**2
             + l**2 / c**2 * sin(ga)**2
             + 2*k*l/b/c * (cos(be)*cos(ga) - cos(al))
             + 2*l*h/c/a * (cos(ga)*cos(al) - cos(be))
             + 2*h*k/a/b * (cos(al)*cos(be) - cos(ga))
        )
    )


def equivhkl(symops, indices, laue=False):
    """Returns list of equivalent Miller indices

    Parameters
    ----------
    symops : list
        list of symmetry operations (4*4 augmented matrices)
    indices : list
        [h, k, l]
    laue : bool
        whether Friedel pairs are considered as equivalents (default False)

    Returns
    -------
    list
        [[h1, k1, l1], [h2, k2, l2], ...]
    """

    result = []
    for i in symops:
        hkl = [sum([indices[j]*i[j][k] for j in range(3)]) for k in range(3)]
        if hkl not in result:
            result.append(hkl)
    if ([-i for i in indices] not in result) and laue:
        result += equivhkl(symops, [-i for i in indices])
    return result


def lattrange(cell, r):
    """Returns lattice half-range covering sphere of given radius

    Parameters
    ----------
    cell : list
        [a, b, c, al, be, ga]
    r : float
        radius

    Returns
    -------
    numpy.ndarray
        [Na, Nb, Nc]
    """

    from numpy import array, ceil

    d1 = array([dhkl(cell, hkl) for hkl in ([1, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 1])])
    return ceil(r / d1).astype(int)


def length(cell, v1, v2=(0, 0, 0, 1),
           cell_esd=(0, 0, 0, 0, 0, 0),
           v1_esd=(0, 0, 0, 0),
           v2_esd=(0, 0, 0, 0),
           skipesd=False):
    """Returns length and its esd

    Parameters
    ----------
    cell : list
        [a, b, c, al, be, ga]
    v1 : list
        first vector in [x, y, z, 1] notation
    v2 : list
        second vector in [x, y, z, 1] notation
        (default (0, 0, 0, 1))
    cell_esd : list
        esds of cell, default zero
    v1_esd : list
        esds of v1 components (default zero)
    v2_esd : list
        esds of v2 components (default zero)
    skipesd : bool
        if True, skips esd evaluation (default False)

    Returns
    -------
    tuple
        length of v1-v2 vector in cell basis
        together with esd
    """

    from math import cos, sin, sqrt, pi

    a, b, c = cell[:3]
    al, be, ga = [i/180*pi for i in cell[3:]]
    u, v, w = [v1[i] - v2[i] for i in range(3)]
    d = sqrt(a**2*u**2 + b**2*v**2 + c**2*w**2
             + 2*b*c*v*w*cos(al) + 2*a*c*u*w*cos(be) + 2*a*b*u*v*cos(ga))

    if (d == 0) or skipesd:
        d_esd = 0
    else:
        a_esd, b_esd, c_esd = cell_esd[:3]
        al_esd, be_esd, ga_esd = [i/180*pi for i in cell_esd[3:]]
        u_esd, v_esd, w_esd = [sqrt(v1_esd[i]**2 + v2_esd[i]**2)
                               for i in range(3)]
        d_esd = sqrt((a**2*b**2*ga_esd**2*u**2*v**2*sin(ga)**2
                      + a**2*be_esd**2*c**2*u**2*w**2*sin(be)**2
                      + a**2*u_esd**2*(a*u + b*v*cos(ga) + c*w*cos(be))**2
                      + a_esd**2*u**2*(a*u + b*v*cos(ga) + c*w*cos(be))**2
                      + al_esd**2*b**2*c**2*v**2*w**2*sin(al)**2
                      + b**2*v_esd**2*(a*u*cos(ga) + b*v + c*w*cos(al))**2
                      + b_esd**2*v**2*(a*u*cos(ga) + b*v + c*w*cos(al))**2
                      + c**2*w_esd**2*(a*u*cos(be) + b*v*cos(al) + c*w)**2
                      + c_esd**2*w**2*(a*u*cos(be) + b*v*cos(al) + c*w)**2)
                     / (a**2*u**2 + 2*a*b*u*v*cos(ga)
                        + 2*a*c*u*w*cos(be) + b**2*v**2
                        + 2*b*c*v*w*cos(al) + c**2*w**2))
    return (d, d_esd)


def matrixform(symop):
    from sympy import parsing, symbols
    # generates augmented 4*4 matrix from _space_group_symop_operation_xyz
    w_aug = []
    x, y, z = symbols("x, y, z")
    for i in symop.lower().replace(" ", "").replace("\t", "").split(","):
        expr = parsing.sympy_parser.parse_expr(i)
        t = expr - expr.coeff(x)*x - expr.coeff(y)*y - expr.coeff(z)*z
        w_aug.append([float(expr.coeff(x)),
                      float(expr.coeff(y)),
                      float(expr.coeff(z)),
                      float(t)])
    w_aug.append([0, 0, 0, 1])
    return w_aug


def newbasis(cell, P, cell_esd=(0, 0, 0, 0, 0, 0)):
    """Returns transformed basis [a, b, c, al, be, ga] and esds

    Parameters
    ----------
    cell : list
        [a, b, c, al, be, ga]
    P : list
        4*4 P-matrix (note column by column order)
    cell_esd : list
        esds of cell parameters

    Returns
    -------
    list
        new [a, b, c, al, be, ga] and esds
    """

    a = [i[0] for i in P[:3]]
    b = [i[1] for i in P[:3]]
    c = [i[2] for i in P[:3]]
    al, al_esd = angle(cell, [0, 0, 0], b, c, cell_esd=cell_esd)
    be, be_esd = angle(cell, [0, 0, 0], c, a, cell_esd=cell_esd)
    ga, ga_esd = angle(cell, [0, 0, 0], a, b, cell_esd=cell_esd)
    a, a_esd = length(cell, a, cell_esd=cell_esd)
    b, b_esd = length(cell, b, cell_esd=cell_esd)
    c, c_esd = length(cell, c, cell_esd=cell_esd)
    return ([a, b, c, al, be, ga],
            [a_esd, b_esd, c_esd, al_esd, be_esd, ga_esd])


def orthonorm(cell, frac):
    from numpy import array, cos, dot, pi, sin
    # converts fractional
    # to absolute coordinates in orthonormal basis XYZ
    # (X || x*, Z || z) after McKie & McKie 1986 (p. 154)
    a, b, c, al, be, ga = [i for i in cell]
    al *= pi/180
    be *= pi/180
    ga *= pi/180
    # below cos and sin of reciprocal gamma* will be calculated:
    cosGA = ((cos(al)*cos(be) - cos(ga))
             / (sin(al)*sin(be)))
    sinGA = (1-cosGA**2)**0.5
    M = [[a*sin(be)*sinGA, 0, 0],
         [-a*sin(be)*cosGA, b*sin(al), 0],
         [a*cos(be), b*cos(al), c]]
    return dot(array(M), array(frac[:3])).tolist()


def parsecif(source, whitelist=whitelist_structure, ignoreloops=False):
    """Returns CIF content

    Parameters
    ----------
    source : file
        file object
    whitelist : list
        list of CIF keys to import (default whitelist_structure)
        if None, all keys will be imported
    ignoreloops : bool
        whether looped keys should be ignored
        (works only with whitelist=None)

    Returns
    -------
    dict
        {"name": [str1, str2, ...],
         "data": [dict1, dict2, ...],
         "loops": [list1, list2, ...]}
        "name" contains list of datablock names
        "data" contains list of datablock contents
        as dictionaries {CIF_key: value, ...}
            - looped keys are converted into values of list type
            - all keys are converted to lowercase
            - all values are of str type
            - keys with whitespace values are ignored
        "loops" contains list of looped keys (as lists)
    """

    from shlex import split

    def group(keys, values):
        """Sequentially splits N*M values into N lists

        Parameters
        ----------
        keys : list
            list of N keys
        values : list
            list of N*M values

        Returns
        -------
        dict
            {key1: [M values], ..., keyN: [M values]},
            or (when M=1) {key1: value, ..., keyN: value}
        """

        result = {}
        if values != []:
            for key in keys:
                result[key] = []
            for j in range(len(values)):
                result[keys[j % len(keys)]].append(values[j])
            if len(values) == 1:
                result[keys[0]] = result[keys[0]][0]
        return result

    if whitelist is not None:
        whitelist = [i.lower() for i in whitelist]
    keys = []
    values = []
    flags = {"loop": False, "values": False, "text": False, "ignore": True}
    parsed = []
    loops = []
    datablocks = []
    file = source
    file.seek(0)
    for line in file:
        if isinstance(line, bytes):
            line = line.decode(errors='ignore')
        if (len(line) == 0) or line.startswith('#'):
            continue
        if line.startswith(";"):
            if not flags["text"]:
                flags["text"] = True
                current = [line[1:line.find(' #')]]
                continue
            else:
                flags["text"] = False
                current = [current[0].rstrip()]
        else:
            if flags["text"]:
                if line.startswith("#"):
                    continue
                else:
                    current = [current[0] + '\n' + line[:line.find(' #')]]
                    continue
            else:
                try:
                    current = split(line, comments=True)
                except ValueError:
                    current = line.split()
        for i in current:
            if i.startswith("data_"):
                if datablocks != []:
                    flags["values"] = False
                    parsed[-1].update(group(keys, values))
                    if (len(keys) > 1) and (flags["ignore"] is False):
                        loops[-1].append(keys)
                    flags["ignore"] = True
                    keys = []
                    values = []
                datablocks.append(i[5:])
                parsed.append({})
                loops.append([])
            elif i.startswith("_") or (i == "loop_"):
                if flags["values"]:
                    flags["values"] = False
                    parsed[-1].update(group(keys, values))
                    if (len(keys) > 1) and (flags["ignore"] is False):
                        loops[-1].append(keys)
                    flags["ignore"] = True
                    keys = []
                    values = []
                if i == "loop_":
                    flags["loop"] = True
                else:
                    if whitelist is None:
                        if not (flags['loop'] and ignoreloops):
                            flags["ignore"] = False
                    elif i.lower() in whitelist:
                        flags["ignore"] = False
                    if flags["loop"]:
                        keys.append(i.lower())
                    else:
                        keys = [i.lower()]
            else:
                flags["values"] = True
                flags["loop"] = False
                # ignores empty multistrings
                if not flags["ignore"] and (i != ''):
                    values.append(i)
    # when file ends with looped values,
    # writes last item into parsed and loops dicts
    parsed[-1].update(group(keys, values))
    if (len(keys) > 1) and (flags["ignore"] is False):
        loops[-1].append(keys)
    # note that loop with single key converts into simple key-value pair
    return {"name": datablocks, "data": parsed, "loops": loops}


def qimport(path):
    """Quick wrapper for parsecif

    Parameters
    ----------
    path : string
        location of cif

    Returns
    -------
    list
        list of Structure instances
    """

    from numpy import array

    with open(path) as f:
        data = parsecif(f)['data']
        structures = array([readstruct(i) for i in data])
    return structures[structures != array(None)]


def readesd(s):
    # returns floats (val, esd) from string "val(esd)"
    val = s.split("(")[0]
    if "(" in s:
        esd = s.split("(")[1][:-1]
    else:
        esd = 0

    if val.find(".") == -1:
        m = 0
    else:
        m = len(val) - 1 - val.find(".")
    try:
        return (float(val), float(esd) / 10**m)
    except ValueError:
        return (None, None)


def readformula(f):
    """Interprets _chemical_formula_sum value

    Parameters
    ----------
    f : str
        value of _chemical_formula_sum key

    Returns
    -------
    dict
        {symbol: content, ...}

    """

    def index(s):
        i = -1
        while True:
            if s[i] in '0123456789.':
                i -= 1
                continue
            else:
                break
        if i == -1:
            return {s: 1.0}
        else:
            return {s[:i+1]: float(s[i+1:])}

    f = f.replace("(", "")
    f = f.replace(")", "")
    items = f.split()
    result = {}
    for i in items:
        result.update(index(i))
    return result


def readstruct(data):
    """Returns Structure instance from CIF-based dict

    Parameters
    ----------
    data : dict
        {key: value, ...}

    Returns
    -------
    Structure
        (empty site list is allowed)
    None
        (when cell metrics is not in data)
    """

    from math import pi

    keys = ["_cell_length_a", "_cell_length_b", "_cell_length_c",
            "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
            "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    data = clearkeys(data)[0]
    if not (set(keys[:6]) <= set(data.keys())):
        return None
    struct = Structure([readesd(data[j])[0] for j in keys[:6]],
                       [readesd(data[j])[1] for j in keys[:6]],
                       [Site([*[readesd(data[k][j])[0]
                              for k in keys[6:]], 1],
                             [*[readesd(data[k][j])[1]
                              for k in keys[6:]], 0])
                        for j in range(len(data[
                            "_atom_site_fract_x"]))
                        ] if '_atom_site_fract_x' in data else [])
    if "_atom_site_label" in data:
        for i in range(len(struct.sites)):
            struct.sites[i].label = \
                data["_atom_site_label"][i]
    if "_atom_site_type_symbol" in data:
        for i in range(len(struct.sites)):
            struct.sites[i].symbol = \
                data["_atom_site_type_symbol"][i]
    else:
        for i in range(len(struct.sites)):
            struct.sites[i].symbol = \
                ''.join([char for char in struct.sites[i].label
                         if char.isalpha()])
    if "_atom_site_occupancy" in data:
        for i in range(len(struct.sites)):
            struct.sites[i].occ, struct.sites[i].occ_esd = \
                readesd(data["_atom_site_occupancy"][i])
    if "_atom_site_u_iso_or_equiv" in data:
        for i in range(len(struct.sites)):
            struct.sites[i].u, struct.sites[i].u_esd = \
                readesd(data["_atom_site_u_iso_or_equiv"][i])
    elif "_atom_site_b_iso_or_equiv" in data:
        for i in range(len(struct.sites)):
            b, b_esd = readesd(data["_atom_site_b_iso_or_equiv"][i])
            struct.sites[i].u = b*3/8/pi**2
            struct.sites[i].u_esd = b_esd*3/8/pi**2
    if "_space_group_symop_operation_xyz" in data:
        if type(data["_space_group_symop_operation_xyz"]) is str:
            struct.symops = [matrixform(
                data["_space_group_symop_operation_xyz"])]
        else:
            struct.symops = [matrixform(j) for j
                             in data["_space_group_symop_operation_xyz"]]
    elif "_symmetry_equiv_pos_as_xyz" in data:
        if type(data["_symmetry_equiv_pos_as_xyz"]) is str:
            struct.symops = [matrixform(
                data["_symmetry_equiv_pos_as_xyz"])]
        else:
            struct.symops = [matrixform(j) for j
                             in data["_symmetry_equiv_pos_as_xyz"]]
    return struct


def stringform(w_aug):
    # converts augmented symop matrix to string notation
    from sympy import nsimplify, symbols

    x, y, z = symbols("x, y, z")
    expr = []
    for i in w_aug[:3]:
        expr.append(nsimplify(i[0]*x + i[1]*y + i[2]*z + i[3]))
    return ",".join([str(i) for i in expr]).replace(" ", "")


def vol(cell, cell_esd=(0, 0, 0, 0, 0, 0)):
    """Returns unit cell volume

    Parameters
    ----------
    cell : list
        [a, b, c, alpha, beta, gamma]
    cell_esd : list
        esds of cell (default zeroes)

    Returns
    -------
    tuple
        (volume, esd)
    """

    from math import cos, sin, sqrt, pi

    a, b, c = cell[:3]
    a_esd, b_esd, c_esd = cell_esd[:3]
    al, be, ga = [i/180*pi for i in cell[3:]]
    al_esd, be_esd, ga_esd = [i/180*pi for i in cell_esd[3:]]
    v = a*b*c*sqrt(sin(al)**2 + sin(be)**2 + sin(ga)**2
                   + 2*cos(al)*cos(be)*cos(ga) - 2)
    v_esd = sqrt(
        (a**2*al_esd**2*b**2*c**2*(cos(al) - cos(be)*cos(ga))**2*sin(al)**2
         + a**2*b**2*be_esd**2*c**2*(cos(al)*cos(ga) - cos(be))**2*sin(be)**2
         + a**2*b**2*c**2*ga_esd**2*(cos(al)*cos(be) - cos(ga))**2*sin(ga)**2
         + (a**2*b**2*c_esd**2 + a**2*b_esd**2*c**2 + a_esd**2*b**2*c**2)
         * (sin(al)**2 + sin(be)**2 + sin(ga)**2
            + 2*cos(al)*cos(be)*cos(ga) - 2)**2)
        / (sin(al)**2 + sin(be)**2 + sin(ga)**2
           + 2*cos(al)*cos(be)*cos(ga) - 2)
    )
    return (v, v_esd)


def writesd(val, esd, f=None):
    """Returns value with esd in parentheses

    Follows recommendations from Schwarzenbach et al.,
    Acta Crystallogr A51, 565–569 (1995)

    Parameters
    ----------
    val : float
        value
    esd : float
        esd
    f : str
        format to use when esd is 0 or None

    Returns
    -------
    str
        value with esd in parentheses
    """

    from math import log10, ceil

    if val is None:
        return None
    if (esd == 0) or (esd is None):
        if f is None:
            return str(val)
        else:
            return format(val, f)
    power = int(log10(esd) // 1)
    base = esd / 10**power
    if base <= 1.9:
        power -= 1
        base = ceil(esd / 10**power)
    else:
        base = ceil(base)
    if power >= 0:
        return str(int(round(val, -power))) + "(" + str(base * 10**power) + ")"
    else:
        val_str = format(float(round(val, -power)), f'.{-power}f')
        return (val_str
                + "0" * (-power - (len(val_str) - 1 - val_str.find(".")))
                + "(" + str(base) + ")")
