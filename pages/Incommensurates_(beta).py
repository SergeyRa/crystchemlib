from core import parsecif, readesd, whitelist_structure
from incomm import readstruct_mod, whitelist_incomm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def parsefile(file):
    parsed = {'datablock': [], 'data': []}
    if file is not None:
        whitelist = whitelist_structure + whitelist_incomm
        fromfile = parsecif(file, whitelist)
        for n, d in zip(fromfile['name'], fromfile['data']):
            if '_cell_modulation_dimension' in d:
                parsed['datablock'].append(n)
                parsed['data'].append(d)
    return parsed


st.text(
    'Supported modulations:\n'
    '\t- harmonics and crenel for occupancy\n'
    '\t- harmonics and Legendre polynomials (in crenel interval)'
    ' for position\n'
    'Loops with site labels and axes must correspond to loops with modulation'
    " parameters; use of multiple apostrophes in site labels (e.g. Si1'')"
    ' is not supported.'
)

file = st.sidebar.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = st.sidebar.selectbox('Choose datablock', parsed['datablock'])
if datablock is not None:
    data = parsed['data'][parsed['datablock'].index(datablock)]
    structure = readstruct_mod(data)
    q = st.sidebar.selectbox('Choose q', range(1, len(structure.q[0])+1))
    for i, m in enumerate(structure.q[0]):
        st.sidebar.text(f'q{i+1}: {str(m)}')
    st.sidebar.download_button('Export average structure',
                               structure.cif(),
                               file_name=file.name[:-4]+'_average.cif')
    na = st.sidebar.number_input('Cells along a:', 1, 10, 1)
    nb = st.sidebar.number_input('Cells along b:', 1, 10, 1)
    nc = st.sidebar.number_input('Cells along c:', 1, 10, 1)
    st.sidebar.download_button('Export approximant',
                               structure.approx((na, nb, nc)).cif(),
                               file_name=file.name[:-4]+f'_{na}x{nb}x{nc}.cif')

    labels = [s.label for s in structure.sites]
    selected = st.selectbox('Choose central site', labels)
    ncentral = labels.index(selected)
    site = structure.sites[ncentral]
    df1 = pd.DataFrame({'x4': np.linspace(0, 1, 1000)})
    for i in range(3):
        df1['xyz'[i]], df1['xyz'[i]+'_esd'] = np.vectorize(
            site.modxyz[i][q-1].val
        )(df1['x4'], zero=False)
    df1['o'], df1['o_esd'] = np.vectorize(
            site.modocc[q-1].val
        )(df1['x4'], zero=False)
    occ, occ_esd = site.occ, site.occ_esd
    if site.modocc[q-1].form == 'cren':
        w = site.modocc[q-1].params[1]
        w_esd = site.modocc[q-1].esds[1]
        occ /= w
        occ_esd = ((occ_esd**2 - w_esd**2 * occ**2) / w**2)**0.5
        df1['o'] = [occ if i == 1 else None for i in df1['o']]
        df1['o_esd'] = [occ_esd if i is not None else None for i in df1['o']]
    elif site.modocc[q-1].form == 'none':
        df1['o'] = occ
        df1['o_esd'] = occ_esd
    else:
        df1['o'] = occ*(1+df1['o'])
        df1['o_esd'] = (
            occ_esd**2 * (1+df1['o'])**2 + df1['o_esd']**2 * occ**2
        )**0.5

    clabel = 'fractional displacement'
    coord = st.toggle('Plot absolute displacements')
    if coord:
        clabel = 'absolute displacement, A'
        a, a_esd = readesd(data['_cell_length_a'])
        b, b_esd = readesd(data['_cell_length_b'])
        c, c_esd = readesd(data['_cell_length_c'])
        df1['x_esd'] = ((df1['x_esd'] * a)**2+(a_esd * df1['x'])**2)**0.5
        df1['y_esd'] = ((df1['y_esd'] * b)**2+(b_esd * df1['y'])**2)**0.5
        df1['z_esd'] = ((df1['z_esd'] * c)**2+(c_esd * df1['z'])**2)**0.5
        df1['x'] = df1['x']*a
        df1['y'] = df1['y']*b
        df1['z'] = df1['z']*c

    col1, col2 = st.columns(2)

    fig1 = go.Figure(layout={'xaxis': {'title': {'text': 'x4'}},
                             'yaxis': {'title': {'text': 'occupancy'}}})
    fig1.add_trace(go.Scatter(x=df1['x4'], y=df1['o']+df1['o_esd'],
                              showlegend=False,
                              line={'dash': 'dot',
                                    'color': 'black'}))
    fig1.add_trace(go.Scatter(x=df1['x4'], y=df1['o']-df1['o_esd'],
                              showlegend=False,
                              line={'dash': 'dot',
                                    'color': 'black'}))
    fig1.add_trace(go.Scatter(x=df1['x4'], y=df1['o'],
                              showlegend=False,
                              line={'color': 'black'}))
    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)
    col1.plotly_chart(fig1)

    fig2 = go.Figure(layout={'xaxis': {'title': {'text': 'x4'}},
                             'yaxis': {'title': {'text': clabel}}})
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    for i in 'xyz':
        fig2.add_trace(go.Scatter(x=df1['x4'], y=df1[i]+df1[i+'_esd'],
                                  showlegend=False,
                                  line={'dash': 'dot',
                                        'color': colors[i]}))
        fig2.add_trace(go.Scatter(x=df1['x4'], y=df1[i]-df1[i+'_esd'],
                                  showlegend=False,
                                  line={'dash': 'dot',
                                        'color': colors[i]}))
        fig2.add_trace(go.Scatter(x=df1['x4'], y=df1[i],
                                  name='d'+i,
                                  line={'color': colors[i]}))
    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)
    col2.plotly_chart(fig2)

    st.download_button(
        'Export curves as CSV-file',
        df1.rename(columns={'o': 'occ', 'o_esd': 'occ_esd'}).to_csv(None),
        file_name=site.label+'_modulation.csv',
        width='stretch'
    )

    ligands = st.multiselect("Choose ligands", labels)
    dmin, dmax = st.slider("Choose bond length range",
                           0.0, 10.0, (0.1, 3.0), 0.1)
    if len(ligands) != 0:
        poly = structure.poly(ncentral, structure.filter('label', ligands),
                              dmax=dmax, dmin=dmin, suffixes=True)
        options = {'distances': 'Distances, A',
                   'angles': 'Angles, deg',
                   'volume': 'Polyhedron volume, A^3'}
        plot = st.selectbox('Select value for t-plot', options.keys())
        poly.q = structure.q
        tsteps = 1000
        T = np.array(
            [np.linspace(0, 1, tsteps) if (i == q-1) else np.zeros(tsteps)
             for i in range(len(structure.q[0]))]
        ).T
        P = [poly.t(t) for t in T]

        dist = [pd.DataFrame(p.listdist()) for p in P]
        for d, t in zip(dist, T[:, q-1]):
            d['t'] = t
            d.set_index(['t', 'name'], inplace=True)
        df2 = pd.concat(dist).unstack().rename(
            columns={'value': 'distances', 'esd': 'distances_esd'}
        )

        angles = [pd.DataFrame(p.listangl()) for p in P]
        for d, t in zip(angles, T[:, q-1]):
            d['t'] = t
            d.set_index(['t', 'name'], inplace=True)
        df2 = pd.concat([
            df2, pd.concat(angles).unstack().rename(
                columns={'value': 'angles', 'esd': 'angles_esd'}
            )
        ], axis='columns')

        (df2[('volume', site.label)],
         df2[('volume_esd', site.label)]) = np.array(
             [p.polyvol() for p in P]
         ).T

        pd.options.plotting.backend = 'plotly'
        fig3 = df2[plot].plot()
        fig3.update_layout(
            yaxis=dict(
                title=dict(
                    text=options[plot]
                )
            ),
            legend=dict(
                title=dict(
                    text=None
                )
            )
        )
        st.plotly_chart(fig3)

        st.download_button(
            'Export curves as CSV-file',
            df2.to_csv(None),
            file_name=site.label+'_polyhedron_modulation.csv',
            width='stretch'
        )
