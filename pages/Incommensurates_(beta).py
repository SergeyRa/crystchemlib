from core import parsecif, readesd, whitelist_structure
from incomm import readmodf, modv, whitelist_incomm
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
    """Supported modulations:
    \t- harmonics and crenel for occupancy
    \t- harmonics and Legendre polynomials (in crenel interval) for position

    Site labels and axes must be present in loops with modulation parameters
    (modulation id keys are not yet supported); use of multiple apostrophes
    in site labels (e.g. Si1'') is not supported."""
)

col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col1.selectbox('Choose datablock', parsed['datablock'])
if datablock is not None:
    data = parsed['data'][parsed['datablock'].index(datablock)]
    modv = modv(data)[0]
    q = col2.selectbox('Choose q', range(1, len(modv)+1))
    for i, m in enumerate(modv):
        col2.text(f'q{i+1}: {str(m)}')
    site = col2.selectbox('Choose site', data['_atom_site_label'])
    df = pd.DataFrame({'x4': np.linspace(0, 1, 1000)})
    for p in 'oxyz':
        df[p], df[p+'_esd'] = np.vectorize(
            readmodf(data, site, p)[q-1].val)(df.x4, zero=False)
    occ, occ_esd = readesd(
        data['_atom_site_occupancy'][data['_atom_site_label'].index(site)]
    )
    if readmodf(data, site, 'o')[q-1].form == 'cren':
        w = readmodf(data, site, 'o')[q-1].params[1]
        w_esd = readmodf(data, site, 'o')[q-1].esds[1]
        occ /= w
        occ_esd = ((occ_esd**2 - w_esd**2 * occ**2) / w**2)**0.5
        df.o = [occ if i == 1 else None for i in df.o]
        df.o_esd = [occ_esd if i is not None else None for i in df.o]
    elif readmodf(data, site, 'o')[q-1].form == 'none':
        df.o = occ
        df.o_esd = occ_esd
    else:
        df.o = occ*(1+df.o)
        df.o_esd = (occ_esd**2 * (1+df.o)**2 + df.o_esd**2 * occ**2)**0.5

    clabel = 'fractional displacement'
    coord = col2.toggle('Plot absolute displacements')
    if coord:
        clabel = 'absolute displacement, A'
        a, a_esd = readesd(data['_cell_length_a'])
        b, b_esd = readesd(data['_cell_length_b'])
        c, c_esd = readesd(data['_cell_length_c'])
        df.x_esd = ((df.x_esd * a)**2+(a_esd * df.x)**2)**0.5
        df.y_esd = ((df.y_esd * b)**2+(b_esd * df.y)**2)**0.5
        df.z_esd = ((df.z_esd * c)**2+(c_esd * df.z)**2)**0.5
        df.x = df.x*a
        df.y = df.y*b
        df.z = df.z*c

    st.divider()

    col3, col4 = st.columns(2)

    fig1 = go.Figure(layout={'xaxis': {'title': {'text': 'x4'}},
                             'yaxis': {'title': {'text': 'occupancy'}}})
    fig1.add_trace(go.Scatter(x=df.x4, y=df.o+df.o_esd,
                              showlegend=False,
                              line={'dash': 'dot',
                                    'color': 'black'}))
    fig1.add_trace(go.Scatter(x=df.x4, y=df.o-df.o_esd,
                              showlegend=False,
                              line={'dash': 'dot',
                                    'color': 'black'}))
    fig1.add_trace(go.Scatter(x=df.x4, y=df.o,
                              showlegend=False,
                              line={'color': 'black'}))
    fig1.update_xaxes(showgrid=True)
    fig1.update_yaxes(showgrid=True)
    col3.plotly_chart(fig1)

    fig2 = go.Figure(layout={'xaxis': {'title': {'text': 'x4'}},
                             'yaxis': {'title': {'text': clabel}}})
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    for i in 'xyz':
        fig2.add_trace(go.Scatter(x=df.x4, y=df[i]+df[i+'_esd'],
                                  showlegend=False,
                                  line={'dash': 'dot',
                                        'color': colors[i]}))
        fig2.add_trace(go.Scatter(x=df.x4, y=df[i]-df[i+'_esd'],
                                  showlegend=False,
                                  line={'dash': 'dot',
                                        'color': colors[i]}))
        fig2.add_trace(go.Scatter(x=df.x4, y=df[i],
                                  name='d'+i,
                                  line={'color': colors[i]}))
    fig2.update_xaxes(showgrid=True)
    fig2.update_yaxes(showgrid=True)
    col4.plotly_chart(fig2)

    st.dataframe(df.rename(columns={'o': 'occ',
                                    'o_esd': 'occ_esd'}))
