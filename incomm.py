# TODO: link loops using id

"""Module for handling incommensurately modulated structures

Classes
-------
Modf
    Defines modulation function

Functions
---------
approx : Structure
    Approximant (monkeypatched as Structure method)
harmcomp : int
    Component of modulation vector in harmonic
modv : tuple or None
    Extracts modulation vectors
x4 : Site
    Defines modulated Site for given x4
    (monkeypatched as Site method)
t : Polyhedron
    Defines modulated Polyhedron for given t
    (monkeypatched as Polyhedron method)
readmodf : list
    Extracts modulation functions
readstruct_mod : core.Structure
    Reads modulated structure from CIF-based dict
"""

from core import Polyhedron, Site, Structure

# CIF keys used
whitelist_incomm = [
    '_atom_site_displace_fourier_atom_site_label',
    '_atom_site_displace_fourier_axis',
    '_atom_site_displace_fourier_param_cos',
    '_atom_site_displace_fourier_param_sin',
    '_atom_site_displace_fourier_wave_vector_seq_id',
    '_atom_site_fourier_wave_vector_seq_id',
    '_atom_site_fourier_wave_vector_x',
    '_atom_site_fourier_wave_vector_y',
    '_atom_site_fourier_wave_vector_z',
    '_atom_site_occ_fourier_atom_site_label',
    '_atom_site_occ_fourier_param_cos',
    '_atom_site_occ_fourier_param_sin',
    '_atom_site_occ_fourier_wave_vector_seq_id',
    '_atom_site_occ_special_func_atom_site_label',
    '_atom_site_occ_special_func_crenel_c',
    '_atom_site_occ_special_func_crenel_w',
    '_cell_modulation_dimension',
    '_cell_wave_vector_seq_id',
    '_cell_wave_vector_x',
    '_cell_wave_vector_y',
    '_cell_wave_vector_z',
    '_jana_atom_site_displace_legendre_atom_site_label',
    '_jana_atom_site_displace_legendre_axis',
    '_jana_atom_site_displace_legendre_param_coeff',
    '_jana_atom_site_displace_legendre_param_order',
    '_jana_atom_site_fourier_wave_vector_q1_coeff',
    '_space_group_symop_ssg_operation_algebraic']


class Modf:
    """Defines modulation function

    Attributes
    ----------
    form : str
        form of modulation function
    params : numpy.ndarray or list
        parameters of modulation function
    esds : numpy.ndarray
        esd of parameters

    Methods
    -------
    ptv : tuple
        Peak-to-valley height
    val : tuple
        Returns modulation function value and esd
    """

    def __init__(self, form, params=None, esds=None):
        """
        Parameters
        ----------
        form : str
            form of modulation function:
            harm - harmonic
            cren - crenel
            lege - Legendre polynomials on crenel interval
            none
        params : numpy.ndarray or list
            parameters of modulation function
        esds : numpy.ndarray
            esds of parameters
        """

        self.form = form
        self.params = params
        self.esds = esds

    def ptv(self):
        """Peak-to-valley height

        Returns
        -------
        tuple
            (ptv, esd)
        """

        from scipy.optimize import minimize_scalar

        x4p = minimize_scalar(lambda x: -self.val(x)[0],
                              bounds=(0, 1), method='bounded').x
        x4v = minimize_scalar(lambda x: self.val(x)[0],
                              bounds=(0, 1), method='bounded').x
        p, p_esd = self.val(x4p)
        v, v_esd = self.val(x4v)
        # TODO: improve esd evaluation:
        return (p-v, (p_esd**2 + v_esd**2)**0.5)

    def val(self, x4, zero=True):
        """Returns modulation function value and esd

        Parameters
        ----------
        x4 : float
            phase of modulation vector
        zero : bool
            when True (default), (0.0, 0.0) is returned
            outside crenel interval and for self.form=='none',
            otherwise (None, None)

        Returns
        -------
        tuple
            modulation function value and esd
        """

        na = (0.0, 0.0) if zero else (None, None)

        if self.form == 'harm':
            from numpy import sin, cos, pi

            return (
                (self.params[:, 1]*cos(2*pi*self.params[:, 0]*x4)
                 + self.params[:, 2]*sin(2*pi*self.params[:, 0]*x4)).sum(),
                ((self.esds[:, 1]*cos(2*pi*self.params[:, 0]*x4))**2
                 + (self.esds[:, 2]
                    * sin(2*pi*self.params[:, 0]*x4))**2).sum()**0.5
            )

        elif self.form == 'cren':
            from numpy import abs, array
            x4 %= 1
            c, w = self.params
            c %= 1
            # possible nearest crenel centers:
            pncc = array([c-1, c, c+1])
            # distance from x4 to the nearest crenel center:
            d = abs(x4-pncc).min()
            return (1.0, 0.0) if (d <= w/2) else na

        elif self.form == 'lege':
            from numpy import abs, argmin, array

            x4 %= 1
            c, w = self.params[1:]
            c %= 1
            # possible nearest crenel centers:
            pncc = array([c-1, c, c+1])
            # number of nearest crenel center in pncc:
            n = argmin(abs(x4-pncc))
            # signed distance to the nearest crenel center:
            d = (x4-pncc)[n]
            if abs(d) <= w/2:
                from scipy.special import legendre

                x = 2*d/w
                return (sum([coeff*legendre(order)(x) for order, coeff in
                             self.params[0][:]]),
                        sum([(esd*legendre(order)(x))**2 for order, esd in
                             zip(self.params[0][:, 0], self.esds[:, 1])])**0.5)
            else:
                return na

        elif self.form == 'none':
            return na


def approx(self, abc, occlim=0.0, T0=None):
    """Approximant

    Parameters
    ----------
    abc : tuple
        (na, nb, nc) unit cells in the approximant
    occlim : float
        sites with occupancy below this limit will be omitted
        (default 0.5)
    T0 : np.ndarray
        phase shift in origin along each modulation vector
        (default None)

    Returns
    -------
    Structure
        approximant with P1 space group
    """

    from copy import deepcopy
    from math import ceil
    from numpy import array, dot, eye

    if T0 is None:
        T0 = array([0]*len(self.q[0]))
    a = deepcopy(self.p1())
    a.symops = [eye(4)]
    for i, n in enumerate(abc):
        n = ceil(n)
        a.cell[i] *= n
        stop = len(a.sites)
        for s in a.sites[:stop]:
            # additional sites from neigbor cells
            # will be added in each dimension:
            for j in [-1]+list(range(1, n+1)):
                newsite = deepcopy(s)
                newsite.fract[i] += j
                a.sites.append(newsite)
    pinned = []
    for i in a.sites:
        pinned.append(
            i.x4([t + dot(q, i.fract[:-1]) for t, q in zip(T0, self.q[0])])
        )
    a.sites = []
    for i in pinned:
        if (
                (0 <= i.fract[0] <= abc[0]) and (0 <= i.fract[1] <= abc[1])
                and (0 <= i.fract[2] <= abc[2]) and (i.occ >= occlim)
        ):
            a.sites.append(i)
            for j, n in enumerate(abc):
                a.sites[-1].fract[j] /= ceil(n)
    for i in ('q', 'Rs'):
        delattr(a, i)

    return a


Structure.approx = approx


def harmcomp(data, id, q):
    """Component of modulation vector in harmonic

    Parameters
    ----------
    data : dict
        generated by core.parsecif()
    id : int
        _atom_site_fourier_wave_vector_seq_id
    q : int
        _cell_wave_vector_seq_id

    Returns
    -------
    int
        component of modulation vector in harmonic
    """

    if '_jana_atom_site_fourier_wave_vector_q1_coeff' in data:
        i = data['_atom_site_fourier_wave_vector_seq_id'].index(str(id))
        return int(data[
            '_jana_atom_site_fourier_wave_vector_q'+str(q)+'_coeff'
        ][i])

    elif '_atom_site_fourier_wave_vector_q1_coeff' in data:
        i = data['_atom_site_fourier_wave_vector_seq_id'].index(str(id))
        return int(data[
            '_atom_site_fourier_wave_vector_q'+str(q)+'_coeff'
        ][i])

    else:
        from numpy.linalg import lstsq

        Q = modv(data)[0].transpose()
        Qp = modv(data, '_atom_site_fourier')[0].transpose()
        return lstsq(Q, Qp)[0].round().astype(int).transpose()[id-1][q-1]


def modv(data, prefix='_cell'):
    """Extracts modulation vectors

    Parameters
    ----------
    data : dict
        generated by core.parsecif()
    prefix : str
        '_cell' or '_atom_site_fourier'

    Returns
    -------
    tuple or None
        (modulation vectors, esds)
        as numpy arrays
    """

    if prefix+'_wave_vector_seq_id' not in data:
        return None
    else:
        from core import readesd
        from numpy import array, isin, vectorize
        from pandas import DataFrame

        readesd_v = vectorize(readesd)
        items = array([prefix+'_wave_vector_'+i for i in 'xyz'])
        index = data[prefix+'_wave_vector_seq_id']
        if len(index) == 1:
            index = [index]
        table = DataFrame(
            {k: data[k] for k in items[isin(items, list(data.keys()))]},
            index=index
        )
        for k in items[isin(items, list(data.keys()), invert=True)]:
            table[k] = '0'

        return readesd_v(array(table[items]))


def x4(self, X):
    """Defines modulated Site for given x4

    Parameters
    ----------
    X : numpy.ndarray
        values of modulation phases [x4, x5, ...]

    Returns
    -------
    Site
        new non-modulated site
    """

    from copy import deepcopy
    from numpy import array, hstack, matmul
    from numpy.linalg import inv

    Xt = matmul(inv(self.modRs), hstack([self.fract[:-1], X, [1]]))[3:3+len(X)]
    displ = matmul(
        self.modRs[[0, 1, 2]][:, [0, 1, 2]],
        [sum([f.val(x)[0] for f, x in zip(self.modxyz[i], Xt)])
         for i in range(3)]
    )
    modsite = deepcopy(self)
    for i in ('modxyz', 'modocc', 'modRs'):
        delattr(modsite, i)

    # marking whether occupational modulation is defined:
    occflag = (array([i.form for i in self.modocc]) != 'none')
    docc = array(
        [f.val(x, zero=False)[0] for f, x in zip(self.modocc, Xt)]
    )[occflag]
    if None in docc:  # outside crenel interval
        modsite.occ = 0
        return modsite  # no positional modulation returned
    else:
        modsite.occ = self.occ*(1+sum(docc))  # TODO: esd
    modsite.fract = array(self.fract) + hstack([displ, [0]])  # TODO: esd

    return modsite


Site.x4 = x4


def t(self, T):
    """Defines modulated Polyhedron for given t

    Assumes presence of self.q attribute (see readstruct_mod)

    Parameters
    ----------
    T : numpy.ndarray
        t-values of modulation phases [t(x4), t(x5), ...]

    Returns
    -------
    Polyhedron
        new non-modulated Polyhedron
    """

    from copy import deepcopy
    from numpy import dot

    modpoly = deepcopy(self)
    modpoly.central = self.central.x4(
        [t + dot(q, self.central.fract[:-1]) for t, q in zip(T, self.q[0])]
    )
    modpoly.ligands = [
        i.x4(
            [t + dot(q, i.fract[:-1]) for t, q in zip(T, self.q[0])]
        )
        for i in self.ligands
    ]

    return modpoly


Polyhedron.t = t


def readmodf(data, label, par):
    """Extracts modulation function

    Parameters
    ----------
    data : dict
        dict from core.parsecif()['data']
    label : str
        site label
    par : str
        x, y, z, o (occ)  # TODO u

    Returns
    -------
    list
        Modf for each q-vector
    """

    from core import readesd
    from numpy import array, vectorize
    from pandas import DataFrame

    readesd_v = vectorize(readesd)
    modf = []

    if '_cell_modulation_dimension' not in data:
        return []

    if par in 'xyz':
        # HARMONIC
        if '_atom_site_displace_fourier_atom_site_label' in data:
            if (label, par) in list(zip(
                    data['_atom_site_displace_fourier_atom_site_label'],
                    data['_atom_site_displace_fourier_axis'])):
                for i in range(int(data['_cell_modulation_dimension'])):
                    table = DataFrame({k: data[k] for k in [
                        '_atom_site_displace_fourier_atom_site_label',
                        '_atom_site_displace_fourier_axis',
                        '_atom_site_displace_fourier_wave_vector_seq_id',
                        '_atom_site_displace_fourier_param_cos',
                        '_atom_site_displace_fourier_param_sin']})
                    table['harmcomp'] = [
                        harmcomp(
                            data, j, i+1
                        ) for j in table[
                            '_atom_site_displace_fourier_wave_vector_seq_id'
                        ].astype(int)
                    ]
                    table['harmcomp'] = table['harmcomp'].astype(str)
                    params, esds = readesd_v(
                        array(
                            table.query(
                                '(_atom_site_displace_fourier_atom_site_label'
                                ' == @label)'
                                ' & '
                                '(_atom_site_displace_fourier_axis'
                                ' == @par)'
                            )[['harmcomp',
                               '_atom_site_displace_fourier_param_cos',
                               '_atom_site_displace_fourier_param_sin']]
                        )
                    )
                    modf.append(Modf('harm', params, esds))

        # LEGENDRE
        if '_jana_atom_site_displace_legendre_atom_site_label' in data:
            if (label, par) in list(zip(
                    data['_jana_atom_site_displace_legendre_atom_site_label'],
                    data['_jana_atom_site_displace_legendre_axis'])):
                table = DataFrame({k: data[k] for k in [
                    '_jana_atom_site_displace_legendre_atom_site_label',
                    '_jana_atom_site_displace_legendre_axis',
                    '_jana_atom_site_displace_legendre_param_order',
                    '_jana_atom_site_displace_legendre_param_coeff']})
                params, esds = readesd_v(
                    array(
                        table.query(
                            '(_jana_atom_site_displace_legendre'
                            '_atom_site_label'
                            ' == @label)'
                            ' & '
                            '(_jana_atom_site_displace_legendre_axis'
                            ' == @par)'
                        )[['_jana_atom_site_displace_legendre_param_order',
                           '_jana_atom_site_displace_legendre_param_coeff']]
                    )
                )
                c, w = readmodf(data, label, 'o')[0].params
                modf.append(Modf('lege', [params, c, w], esds))

        # NO POSITIONAL MODULATION
        if modf == []:
            for i in range(int(data['_cell_modulation_dimension'])):
                modf.append(Modf('none'))

    elif par == 'o':
        # HARMONIC
        if '_atom_site_occ_fourier_atom_site_label' in data:
            if label in data['_atom_site_occ_fourier_atom_site_label']:
                for i in range(int(data['_cell_modulation_dimension'])):
                    table = DataFrame({k: data[k] for k in [
                        '_atom_site_occ_fourier_atom_site_label',
                        '_atom_site_occ_fourier_wave_vector_seq_id',
                        '_atom_site_occ_fourier_param_cos',
                        '_atom_site_occ_fourier_param_sin']})
                    table['harmcomp'] = [
                        harmcomp(
                            data, j, i+1
                        ) for j in table[
                            '_atom_site_occ_fourier_wave_vector_seq_id'
                        ].astype(int)
                    ]
                    table['harmcomp'] = table['harmcomp'].astype(str)
                    params, esds = readesd_v(
                        array(
                            table.query(
                                '_atom_site_occ_fourier_atom_site_label'
                                ' == @label'
                            )[['harmcomp',
                               '_atom_site_occ_fourier_param_cos',
                               '_atom_site_occ_fourier_param_sin']]
                        )
                    )
                    modf.append(Modf('harm', params, esds))

        # CRENEL
        if '_atom_site_occ_special_func_atom_site_label' in data:
            if label in data['_atom_site_occ_special_func_atom_site_label']:
                table = DataFrame({k: data[k] for k in [
                    '_atom_site_occ_special_func_atom_site_label',
                    '_atom_site_occ_special_func_crenel_c',
                    '_atom_site_occ_special_func_crenel_w']})
                params, esds = readesd_v(
                    array(
                        table.query(
                            '_atom_site_occ_special_func_atom_site_label'
                            ' == @label'
                        )[['_atom_site_occ_special_func_crenel_c',
                           '_atom_site_occ_special_func_crenel_w']]
                    )[0]
                )
                modf.append(Modf('cren', params, esds))

        # NO OCCUPANCY MODULATION
        if modf == []:
            for i in range(int(data['_cell_modulation_dimension'])):
                modf.append(Modf('none'))

    return modf


def readstruct_mod(data):
    """Reads modulated structure from CIF-based dict

    Parameters
    ----------
    data : dict
        {key: value, ...}

    Returns
    -------
    core.Structure
        with extra attributes
    """

    from core import matrixform, readstruct
    from numpy import eye

    result = readstruct(data)
    result.q = modv(data)
    if '_space_group_symop_ssg_operation_algebraic' in data:
        result.Rs = []
        result.symops = []
        for i in data['_space_group_symop_ssg_operation_algebraic']:
            result.Rs.append(matrixform(i, 3+len(result.q[0])))
            result.symops.append(
                result.Rs[-1][[0, 1, 2, -1]][:, [0, 1, 2, -1]]
            )
    for i in result.sites:
        i.modxyz = [readmodf(data, i.label, j) for j in 'xyz']
        i.modocc = readmodf(data, i.label, 'o')
        i.modRs = eye(3+len(result.q[0])+1)

    return result
