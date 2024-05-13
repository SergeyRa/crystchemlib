import core as ccl
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def parsefiles(files, fullparsing=False):
    parsed = {'source': [], 'structure': [], 'data': [],
              'loops': [], 'poly': []}
    addkeys = ['_cell_volume',
               "_cell_measurement_temperature",
               "_cell_measurement_pressure",
               "_diffrn_ambient_temperature",
               "_diffrn_ambient_pressure",
               '_chemical_formula_sum']
    if fullparsing:
        whitelist = None
    else:
        whitelist = ccl.whitelist_structure + addkeys
    for file in files:
        fromfile = ccl.parsecif(file, whitelist)
        for n, d, lp in zip(fromfile['name'], fromfile['data'],
                            fromfile['loops']):
            d, lp = ccl.clearkeys(d, lp)
            s = ccl.readstruct(d)
            parsed['source'].append(f'{file.name} ({n})')
            parsed['structure'].append(s)
            parsed['data'].append(d)
            parsed['loops'].append(lp)
    return parsed


@st.cache_data
def findpoly(_structures, centrals, ligands, dmin, dmax, nmax):
    """Finds specified polyhedra in list of Structure instances"""

    result = []
    for i in _structures:
        p = []
        if i is None:
            pass
        else:
            lig = i.filter('label', ligands)
            for j in centrals:
                c = i.filter('label', [j])
                if c != []:
                    p.append(i.poly(c[0], lig, dmax, dmin, nmax))
        result.append(p)
    return result


# Functions on datablocks (single-value):
def cifkey(parsed, key):
    """Numerical value of CIF key"""

    result = []
    for d, s in zip(parsed['data'], parsed['source']):
        if key in d.keys():
            result.append([(s, key, *ccl.readesd(d[key]))])
        else:
            result.append([(s, key, None, None)])
    return result


def formcont(parsed, symbol):
    """symbol content from _chemical_formula_sum"""

    result = []
    for d, s in zip(parsed['data'], parsed['source']):
        if '_chemical_formula_sum' in d.keys():
            formula = ccl.readformula(d['_chemical_formula_sum'])
            if symbol in formula.keys():
                result.append([(s, symbol+'sum', formula[symbol], 0)])
            else:
                result.append([(s, symbol+'sum', 0, 0)])
        else:
            result.append([(s, symbol+'sum', None, None)])
    return result


def pgpa(parsed):
    """Pressure in GPa"""

    cell = [i[0] for i in cifkey(parsed, '_cell_measurement_pressure')]
    diffrn = [i[0] for i in cifkey(parsed, '_diffrn_ambient_pressure')]
    result = []
    for c, d, s in zip(cell, diffrn, parsed['source']):
        if c[2] is not None:
            result.append([(s, 'P, GPa', c[2]/1e6, c[3]/1e6)])
        elif d[2] is not None:
            result.append([(s, 'P, GPa', d[2]/1e6, d[3]/1e6)])
        else:
            result.append([(s, 'P, GPa', None, None)])
    return result


def tcelsius(parsed):
    """Temperature in deg Celsius"""

    cell = [i[0] for i in cifkey(parsed, '_cell_measurement_temperature')]
    diffrn = [i[0] for i in cifkey(parsed, '_diffrn_ambient_temperature')]
    result = []
    for c, d, s in zip(cell, diffrn, parsed['source']):
        if c[2] is not None:
            result.append([(s, 'T, deg C', c[2]-273.15, c[3])])
        elif d[2] is not None:
            result.append([(s, 'T, deg C', d[2]-273.15, d[3])])
        else:
            result.append([(s, 'T, deg C', None, None)])
    return result


# Functions on datablocks (multivalue):
def cifloop(parsed, key, label):
    """Numerical values of looped CIF key"""

    result = []
    for d, s in zip(parsed['data'], parsed['source']):
        if (key in d.keys()) and (label in d.keys()):
            tmp = []
            for k, lb in zip(d[key], d[label]):
                tmp.append((s, lb, *ccl.readesd(k)))
            result.append(tmp)
        else:
            result.append([(s, None, None, None)])
    return result


# Functions on polyhedra (single-value):
def cn(parsed):
    """Number of ligands"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            tmp.append((s, i.central.label, len(i.ligands), 0))
        result.append(tmp)
    return result


def econ(parsed):
    """Effective coordination number"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            tmp.append((s, i.central.label, *i.econ()))
        result.append(tmp)
    return result


def hidden_lig(parsed):
    """Number of ligands inside convex hull"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            tmp.append((s, i.central.label, i.hidden(), 0))
        result.append(tmp)
    return result


def occup(parsed):
    """Occupancy of central site"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for j in p:
            tmp.append((s, j.central.label,
                        j.central.occ, j.central.occ_esd))
        result.append(tmp)
    return result


def volume(parsed):
    """Volume"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            tmp.append((s, i.central.label, *i.polyvol()))
        result.append(tmp)
    return result


def volume_corr(parsed):
    """Volume corrected for thermal motion"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            tmp.append((s, i.central.label, *i.polyvol_corr()))
        result.append(tmp)
    return result


# Functions on polyhedra (multi-value):
def angles(parsed):
    """Angles"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            dlist = i.listangl()
            for j, n in enumerate(dlist['name']):
                tmp.append((s, n, dlist['value'][j], dlist['esd'][j]))
        result.append(tmp)
    return result


def bond_weights(parsed):
    """Bond weights"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            dlist = i.bondweights()
            for j, n in enumerate(dlist['name']):
                tmp.append((s, n, dlist['value'][j], dlist['esd'][j]))
        result.append(tmp)
    return result


def distances(parsed):
    """Distances"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            dlist = i.listdist()
            for j, n in enumerate(dlist['name']):
                tmp.append((s, n, dlist['value'][j], dlist['esd'][j]))
        result.append(tmp)
    return result


def distances_corr(parsed):
    """Distances corrected for thermal motion"""

    result = []
    for p, s in zip(parsed['poly'], parsed['source']):
        tmp = []
        for i in p:
            dlist = i.listdist_corr()
            for j, n in enumerate(dlist['name']):
                tmp.append((s, n, dlist['value'][j], dlist['esd'][j]))
        result.append(tmp)
    return result


files = st.file_uploader("Choose CIF file(s)", accept_multiple_files=True)
fullparsing = st.toggle('Read all CIF keys (may be time and memory consuming)')
parsed = parsefiles(files, fullparsing)
st.text(f"{len(parsed['data'])} datablock(s) and "
        f"{len(list(filter(lambda x: x is not None, parsed['structure'])))}"
        f" structure(s) were read from {len(files)} file(s)")

st.divider()
col1, col2, col3, col4 = st.columns(4)
virtsite = col1.toggle('Activate virtual site:')
virtx = col2.number_input('x', min_value=0.0, max_value=1.0)
virty = col3.number_input('y', min_value=0.0, max_value=1.0)
virtz = col4.number_input('z', min_value=0.0, max_value=1.0)
if virtsite:
    for i in parsed['structure']:
        if i is not None:
            i.sites.append(ccl.Site([virtx, virty, virtz, 0],
                                    label='Virtual site'))
labels = set([])
for i in parsed['structure']:
    if i is not None:
        labels = labels.union(set([j.label for j in i.sites]))

chemistry = set([])
for i in parsed['data']:
    if '_chemical_formula_sum' in i.keys():
        chemistry = chemistry.union(
            set(ccl.readformula(i['_chemical_formula_sum']).keys()))
fchem = {}  # functions on formula sum
for i in chemistry:
    fchem[i+' sum'] = lambda x, symbol=i: formcont(x, symbol)

allkeys = set([])
for i in parsed['data']:
    if i is not None:
        allkeys = allkeys.union(set(i.keys()))
loopkeys = set([])
for i in parsed['loops']:
    for j in i:
        loopkeys = loopkeys.union(set(j))
allkeys = allkeys.difference(loopkeys)

fkey = {}  # functions on CIF keys
for i in allkeys:
    fkey[i] = lambda x, key=i: cifkey(x, key)
floop = {}  # dummy functions for looped CIF keys
for i in loopkeys:
    floop[i] = None

col5, col6 = st.columns(2)
dmin, dmax = col5.slider("Choose bond length range",
                         0.0, 10.0, (0.1, 3.0), 0.1)
nmax = col6.slider("Choose coordination limit", 1, 18, 16, 1)
centrals = col5.multiselect("Choose central site",
                            sorted(list(labels)))
ligands = col6.multiselect("Choose ligands",
                           sorted(list(labels)))

# functions with single value output:
fsingle = {'T, C': tcelsius, 'P, GPa': pgpa,
           'Central site occupancy': occup,
           'Effective coordination number': econ,
           'Coordination number': cn,
           'Number of hidden ligands': hidden_lig,
           'Polyhedron volume, A^3': volume,
           'Polyhedron volume (corr.), A^3': volume_corr}
# functions with multivalue output:
fmulti = {'Polyhedron angles, deg': angles,
          'Polyhedron bond weights': bond_weights,
          'Polyhedron distances, A': distances,
          'Polyhedron distances (corr.), A': distances_corr}

xopt = {'CIF keys': fkey, 'Formula content': fchem, 'Other': fsingle}
yopt = {'CIF keys': fkey, 'Formula content': fchem, 'CIF loops': floop,
        'Other': {**fsingle, **fmulti}}

st.divider()
col3, col4 = st.columns(2)
tx = col3.selectbox("Choose X variable type", sorted(list(xopt.keys())))
fx = col3.selectbox("Choose X variable", sorted(list(xopt[tx].keys())))
ty = col4.selectbox("Choose Y variable type", sorted(list(yopt.keys())))
fy = col4.selectbox("Choose Y variable", sorted(list(yopt[ty].keys())))
if ty == 'CIF loops':
    yloop = set([])  # keys from the selected loop
    for i in parsed['loops']:
        for j in i:
            if fy in j:
                yloop = yloop.union(j)
    lb = col4.selectbox("Choose loop label key", sorted(list(yloop)))
    yopt[ty][fy] = lambda x, key=fy, label=lb: cifloop(x, key, label)

if st.button('Run', type="primary", use_container_width=True):
    parsed['poly'] = findpoly(parsed['structure'], centrals, ligands,
                              dmin, dmax, nmax)
    xdata = xopt[tx][fx](parsed) if (fx is not None) else []
    ydata = yopt[ty][fy](parsed) if (fy is not None) else []
    hdata = hidden_lig(parsed)  # check for hidden ligands
else:
    xdata = []
    ydata = []
    hdata = []

x1d = []
y1d = []

xwarn = False
for x, y in zip(xdata, ydata):
    if x != []:
        if len(x) == 1:
            for y1 in y:
                x1d.append(x[0])
                y1d.append(y1)
        elif (len(x) == len(y)) and (fy in fsingle.keys()):
            for x1, y1 in zip(x, y):
                x1d.append(x1)
                y1d.append(y1)
        else:
            xwarn = True
hwarn = False
for h in hdata:
    for h1 in h:
        if h1[2] > 1:
            hwarn = True
if xwarn:
    st.text('Warning: selected Y variable is not supported when '
            '>1 central sites are selected')
    x1d = []
    y1d = []
    hwarn = False  # not necessary
if hwarn:
    st.text('Warning: hidden ligands found!')

df = {'source': [], 'name': [],
      'x': [], 'esd_x': [], 'str_x': [],
      'y': [], 'esd_y': [], 'str_y': []}
for x, y in zip(x1d, y1d):
    df['x'].append(x[2])
    df['esd_x'].append(x[3])
    df['str_x'].append(ccl.writesd(*x[2:]))
    df['source'].append(y[0])
    df['name'].append(y[1])
    df['y'].append(y[2])
    df['esd_y'].append(y[3])
    df['str_y'].append(ccl.writesd(*y[2:]))
df = pd.DataFrame(df)

if len(set(df['name'])) > 24:
    st.text('Warning: duplicate colors in legend!')

if len(df['source']) != 0:
    if (set(df['x']) != {None}) and (set(df['y']) != {None}):
        fig = px.scatter(
            df, x='x', y='y', color='name',
            error_x='esd_x', error_y='esd_y',
            labels={'x': fx, 'y': fy},
            color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(legend={'title_text': ''})
        if fy in fmulti.keys():
            fig.update_layout(legend_traceorder="reversed")
        st.divider()
        st.plotly_chart(fig)
    st.divider()
    df
