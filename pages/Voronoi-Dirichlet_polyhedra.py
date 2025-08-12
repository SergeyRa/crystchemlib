import core as ccl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def hash_func(obj):
    return obj.cif()


@st.cache_data
def parsefile(file):
    parsed = {'datablock': [], 'structure': []}
    if file is not None:
        fromfile = ccl.parsecif(file)
        for n, d in zip(fromfile['name'], fromfile['data']):
            s = ccl.readstruct(d)
            if s is not None:
                parsed['datablock'].append(n)
                parsed['structure'].append(s)
    return parsed


col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col1.selectbox('Choose datablock', parsed['datablock'])

structure = None
for d, s in zip(parsed['datablock'], parsed['structure']):
    if d == datablock:
        structure = s
if structure is None:
    labels = []
else:
    labels = sorted([i.label for i in structure.sites])

cs = col1.selectbox('Central site', labels)
es = col1.multiselect('Excluded sites', labels)
indirect = col1.toggle('Plot indirect neigbors')
uc = col1.toggle('Show basis')
run = st.button('Run', type="primary", use_container_width=True)

if (cs is not None) and run:
    v = structure.voronoi(structure.filter('label', [cs])[0],
                          structure.filter('label', es, inverse=True),
                          wmin=0.001)
    p = v['coordination']
    dist = p.listdist()
    disttxt = np.array([format(i, '.2f') for i in dist['value']])
    cen = ccl.orthonorm(p.cell, p.central.fract)
    lig = np.array([ccl.orthonorm(p.cell, i.fract) for i in p.ligands])
    lab = np.array([i.label for i in p.ligands])
    fig = go.Figure(data=[
        go.Mesh3d(x=v['vertices'][:, 0],
                  y=v['vertices'][:, 1],
                  z=v['vertices'][:, 2],
                  alphahull=0,
                  hoverinfo='none',
                  opacity=0.75)
    ])
    for i in v['direct_neighbors']:
        fig.add_trace(go.Scatter3d(x=[lig[i][0]],
                                   y=[lig[i][1]],
                                   z=[lig[i][2]],
                                   text=lab[i],
                                   hoverinfo='none',
                                   mode='markers+text',
                                   showlegend=False,
                                   marker={'color': 'tomato'}))
        fig.add_trace(go.Scatter3d(x=[cen[0], lig[i][0]],
                                   y=[cen[1], lig[i][1]],
                                   z=[cen[2], lig[i][2]],
                                   text=disttxt[i],
                                   hoverinfo='text',
                                   mode='lines',
                                   showlegend=False,
                                   line={'color': 'tomato'}))
    ind = [i for i in range(len(lig)) if i not in v['direct_neighbors']]
    if indirect:
        for i in ind:
            fig.add_trace(go.Scatter3d(x=[lig[i][0]],
                                       y=[lig[i][1]],
                                       z=[lig[i][2]],
                                       text=lab[i],
                                       hoverinfo='none',
                                       mode='markers+text',
                                       showlegend=False,
                                       marker={'color': 'lightgreen'}))
            fig.add_trace(go.Scatter3d(x=[cen[0], lig[i][0]],
                                       y=[cen[1], lig[i][1]],
                                       z=[cen[2], lig[i][2]],
                                       text=disttxt[i],
                                       hoverinfo='text',
                                       mode='lines',
                                       showlegend=False,
                                       line={'color': 'lightgreen'}))
    if uc:
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        points_on = np.array([ccl.orthonorm(p.cell, i) for i in points])
        fig.add_trace(go.Scatter3d(x=points_on[[0, 1]][:, 0],
                                   y=points_on[[0, 1]][:, 1],
                                   z=points_on[[0, 1]][:, 2],
                                   text='a',
                                   textfont={'color': 'red'},
                                   hoverinfo='text',
                                   mode='lines+text',
                                   showlegend=False,
                                   line={'color': 'red'}))
        fig.add_trace(go.Scatter3d(x=points_on[[0, 2]][:, 0],
                                   y=points_on[[0, 2]][:, 1],
                                   z=points_on[[0, 2]][:, 2],
                                   text='b',
                                   textfont={'color': 'green'},
                                   hoverinfo='none',
                                   mode='lines+text',
                                   showlegend=False,
                                   line={'color': 'green'}))
        fig.add_trace(go.Scatter3d(x=points_on[[0, 3]][:, 0],
                                   y=points_on[[0, 3]][:, 1],
                                   z=points_on[[0, 3]][:, 2],
                                   text='c',
                                   textfont={'color': 'blue'},
                                   hoverinfo='none',
                                   mode='lines+text',
                                   showlegend=False,
                                   line={'color': 'blue'}))
    for i in v['faces']:
        seq = list(range(len(i))) + [0]
        fig.add_trace(go.Scatter3d(x=i[seq][:, 0],
                                   y=i[seq][:, 1],
                                   z=i[seq][:, 2],
                                   hoverinfo='none',
                                   mode='lines',
                                   showlegend=False,
                                   line={'color': 'black'}))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    col2.plotly_chart(fig)
    # O'Keeffe bond weights:
    okbw = np.array(v['weight']) / max(v['weight'])
    # Hoppe bond weights:
    hbw = p.bondweights()['value']
    col2.text(f"Direct neighbors: {len(v['direct_neighbors'])}\n"
              'Indirect neighbors:'
              f" {len(p.ligands) - len(v['direct_neighbors'])}\n"
              f'Voronoi-Dirichlet volume (cubic angstroms):'
              f" {v['volume'][0]:.1f}\n"
              f"O'Keeffe coordination number:"
              f' {okbw.sum():.1f}\n'
              f'Hoppe coordination number (ECoN): {sum(hbw):.1f}')
    df = pd.DataFrame({
        'Neighbor*': lab,
        'Distance (angstroms)': disttxt,
        'Solid angle weight': [format(i, '.3f') for i in v['weight']],
        "O'Keeffe bond weight [1]": [format(i, '.2f') for i in okbw],
        'Hoppe bond weight [2]': [format(i, '.2f') for i in hbw]
    }, index=range(1, len(lab)+1))
    st.dataframe(df)
    st.text('*Indirect neighbors are marked with star\n'
            "[1] O'Keeffe. 1979. Acta Crystallographica A35: 772\n"
            '[2] Hoppe. 1979. Zeitschrift Fur Kristallographie'
            ' - Crystalline Materials 150: 23')
