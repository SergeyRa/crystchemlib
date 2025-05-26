"""Module for handling equation of state

Classes
-------
Eos
    Equation of state

Functions
---------
pvguess : list
    Guesses starting parameters for PV-EoS
pvfit : tuple
    Fits EoS to PV-data
"""

# labels and parameter names:
eoslist = {'BM2': ['2nd order Birch-Murnaghan', 'V0', 'K0'],
           'BM3': ['3rd order Birch-Murnaghan', 'V0', 'K0', 'Kp'],
           'BM4': ['4th order Birch-Murnaghan', 'V0', 'K0', 'Kp', 'Kpp']}


class Eos:
    """Equation of state

    Attributes
    ----------
    kind : str
        kind of eos
    params : list
        parameters of eos
    esds : list
        parameters' esds

    Methods
    -------
    fu : Function
        Returns eos as Function instance
    info : str
        Returns summary on EoS
    """

    def __init__(self, kind, params=None, esds=None):
        """
        Parameters
        ----------
        kind : str
            kind of eos
        params : list
            parameters of eos, default None
        esds : list
            parameters' esds, default None
        """

        self.kind = kind
        self.params = params
        self.esds = esds

    def fu(self, general=False):
        """Returns eos as Function instance

        Parameters
        ----------
        general : bool
            if True, list with eos parameters
            is expected as first argument
        """

        # 4-nd order Birch-Murnaghan
        if self.kind == 'BM4':
            def eos(B, v):
                v0, k0, kp, kpp = B
                f = ((v0/v)**(2/3) - 1) / 2
                return (
                    3*k0*f * (1+2*f)**(5/2)
                    * (1 + 3/2*(kp-4)*f + 3/2*(k0*kpp
                                               + (kp-4)*(kp-3)
                                               + 35/9)*f**2)
                )

        # 3-nd order Birch-Murnaghan
        elif self.kind == 'BM3':
            def eos(B, v):
                v0, k0, kp = B
                f = ((v0/v)**(2/3) - 1) / 2
                return (
                    3*k0*f * (1+2*f)**(5/2)
                    * (1 + 3/2*(kp-4)*f)
                )

        # 2-nd order Birch-Murnaghan
        elif self.kind == 'BM2':
            def eos(B, v):
                v0, k0 = B
                f = ((v0/v)**(2/3) - 1) / 2
                return (
                    3*k0*f * (1+2*f)**(5/2)
                )

        else:
            raise Exception('EoS type unknown for Eos!')

        if general:
            return eos
        else:
            return lambda v: eos(self.params, v)

    def info(self):
        from core import writesd

        text = f'{eoslist[self.kind][0]} equation of state:\n'
        for p, esd, name in zip(self.params, self.esds,
                                eoslist[self.kind][1:]):
            text += f'{name} = {writesd(p, esd)}\n'
        return text


def pvguess(kind, P, V):
    """Guesses starting parameters for PV-EoS

    Parameters
    ----------
    kind : string
        EoS type
    P : np.array
        pressure
    V : np.array
        volume

    Returns
    -------
    list
        starting parameters for EoS fitting
    """

    if kind in ('BM2', 'BM3', 'BM4'):
        v0 = V.max()
        f = ((v0/V)**(2/3) - 1) / 2  # approx. Eulerian strain
        # approx. normalized pressure excluding points with zero strain:
        F = P[f != 0] / (3 * f[f != 0] * (1+2*f[f != 0])**(5/2))
        k0 = F.mean()
        B = [v0, k0]
        if kind in ('BM3', 'BM4'):
            B = pvfit('BM2', P, V)[0].params[:]
            B.append(4.)
            if kind == 'BM4':
                B = pvfit('BM3', P, V)[0].params[:]
                k0, kp = B[1:]
                B.append(-1/k0 * ((3-kp)*(4-kp) + 35/9))

    else:
        raise Exception('EoS type unknown for pvguess!')

    return B


def pvfit(kind, P, V, P_esd=None, V_esd=None, guess=None):
    """Fits EoS to PV-data

    Parameters
    ----------
    P : numpy.array
        pressure
    V : numpy.array
        volume
    P_esd : numpy.array
        pressure esd
    V_esd : numpy.array
        volume esd
    guess : numpy.array
        starting parameters

    Returns
    -------
    tuple
        (Eos, warnings, residual variance)
    """

    from numpy import asarray, isin, isnan
    from scipy import odr

    # convertation to numpy.array:
    P, V = (asarray(P), asarray(V))
    mask = ~(isnan(P) & isnan(V))
    P = P[mask]
    V = V[mask]
    if P_esd is not None:
        P_esd = asarray(P_esd)[mask]
    if V_esd is not None:
        V_esd = asarray(V_esd)[mask]

    warn = ''
    if True in isin(P_esd, (0, 0.0, None)):
        P_esd = None
        warn += 'Zero values in P esd, P weighting ignored!'
    if True in isin(V_esd, (0, 0.0, None)):
        V_esd = None
        warn += '\nZero values in V esd, V weighting ignored!'

    if guess is None:
        guess = pvguess(kind, P, V)
    myf = odr.Model(Eos(kind).fu(general=True))
    mydata = odr.RealData(V, P, V_esd, P_esd)
    myodr = odr.ODR(mydata, myf, guess)
    fit = myodr.run()
    fit.pprint()

    return Eos(kind, list(fit.beta), list(fit.sd_beta)), warn, fit.res_var
