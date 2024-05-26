from scipy.cluster.hierarchy import dendrogram
from nets import Structure, spreadbasis
import core as ccl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def hash_func(obj):
    return obj.cif()


@st.cache_data(hash_funcs={Structure: hash_func})
def densenets(structure, indices):
    return structure.densenets(indices)


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


@st.cache_data(hash_funcs={Structure: hash_func})
def searchnets(structure, groups, maxhkl):
    if groups[0] != []:
        df = structure.searchnets(groups[0], hklmax=[maxhkl]*3)
        df.rename(columns={'mean': 'mean d_eff (A), 1/A^2',
                           'max': 'max d_eff (A), 1/A^2'},
                  inplace=True)
    if groups[1] != []:
        if groups[0] == []:
            df = structure.searchnets(groups[1], hklmax=[maxhkl]*3)
            df.rename(columns={'mean': 'mean d_eff (B), 1/A^2',
                               'max': 'max d_eff (B), 1/A^2'},
                      inplace=True)
        else:
            dfB = structure.searchnets(groups[1], hklmax=[maxhkl]*3)
            dfB.rename(columns={'mean': 'mean d_eff (B), 1/A^2',
                                'max': 'max d_eff (B), 1/A^2'},
                       inplace=True)
            df = df.merge(dfB, on='hkl')
    if (('mean d_eff (B), 1/A^2' in df)
            and ('mean d_eff (B), 1/A^2' in df)):
        df_sum = structure.searchnets(groups[0]+groups[1],
                                      hklmax=[maxhkl]*3)
        df_sum.rename(columns={'mean': 'mean d_eff (A+B), 1/A^2',
                               'max': 'max d_eff (A+B), 1/A^2'},
                      inplace=True)
        df = df.merge(df_sum, on='hkl')
        A = len(structure.sublatt(groups[0]).p1().sites)
        B = len(structure.sublatt(groups[1]).p1().sites)
        df['mean d_eff, 1/A^2'] = ((df['mean d_eff (A), 1/A^2']*A
                                    + df['mean d_eff (B), 1/A^2']*B)
                                   / (A+B))
        df['k_mix'] = ((df['mean d_eff (A+B), 1/A^2']
                        - df['mean d_eff, 1/A^2'])
                       / (df['mean d_eff (A), 1/A^2']
                          + df['mean d_eff (B), 1/A^2']
                          - df['mean d_eff, 1/A^2']))
    df['equivalent hkl for Laue class'] = [ccl.equivhkl(structure.symops,
                                                        i, laue=True)
                                           for i in df['hkl']]
    return df.sort_values(by='mean d_eff, 1/A^2', ascending=False)


for i in ['groups', 'maxhkl', 'result']:
    if i not in st.session_state:
        st.session_state[i] = None
if 'structure' not in st.session_state:
    st.session_state['structure'] = Structure()

col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col2.selectbox('Choose datablock', parsed['datablock'])
maxhkl = col2.slider('Choose max h, k, l', 1, 6, 3)

structure = None
for d, s in zip(parsed['datablock'], parsed['structure']):
    if d == datablock:
        structure = s
if structure is None:
    labels = []
else:
    labels = sorted([i.label for i in structure.sites])

st.divider()
colA, colB = st.columns(2)
slA = colA.multiselect('Sublattice A', labels)
slB = colB.multiselect('Sublattice B', labels)
run = st.button('Run', type="primary", use_container_width=True)
ready = False

if (slA != []) or (slB != []):
    filtA = structure.filter('label', slA)
    filtB = structure.filter('label', slB)
    if (st.session_state['structure'].cif() == structure.cif()
            and st.session_state['groups'] == [filtA, filtB]
            and st.session_state['maxhkl'] == maxhkl):
        result = st.session_state['result']
        ready = True
    elif run:
        st.session_state['structure'] = structure
        st.session_state['groups'] = [filtA, filtB]
        st.session_state['maxhkl'] = maxhkl
        result = searchnets(structure, [filtA, filtB], maxhkl)
        st.session_state['result'] = result
        ready = True

if ready:
    st.divider()
    st.dataframe(result, hide_index=True)
    indices = st.selectbox('Choose hkl from table', result['hkl'])

    st.divider()
    colA1, colB1 = st.columns(2)

    colA1.text('Sublattice A')
    cifA = ""
    if slA != []:
        structA = structure.sublatt(filtA)
        ZA = structA.hklclust(indices)
        if len(ZA) > 0:
            figA, axA = plt.subplots()
            tA = ccl.dhkl(structA.cell, indices) / len(structA.p1().sites)
            # clustering threshold
            dendrogram(ZA, color_threshold=tA,
                       labels=[i.label for i in structA.p1().sites],
                       leaf_rotation=-90)
            axA.hlines(tA, axA.viewLim.bounds[0],
                       axA.viewLim.bounds[0] + axA.viewLim.bounds[2],
                       linestyles='dashed')
            colA1.pyplot(figA)
        else:
            colA1.text('Primitive')
        netsA = densenets(structA, indices)
        dfA = pd.DataFrame()
        dfA['net'] = [
            [structA.p1().sites[j].label for j in i] for i in netsA['cluster']]
        dfA['width, A'] = netsA['width'][:]
        dfA['d_eff, 1/A^2'] = netsA['density'][:]
        colA1.dataframe(dfA)
        for i, s in enumerate(netsA['structure']):
            cifA += s.cif(f"A{i+1}") + '\n\n'

    colB1.text('Sublattice B')
    cifB = ""
    if slB != []:
        structB = structure.sublatt(filtB)
        ZB = structB.hklclust(indices)
        if len(ZB) > 0:
            figB = plt.figure()
            tB = ccl.dhkl(structB.cell, indices) / len(structB.p1().sites)
            # clustering threshold
            dendrogram(ZB, color_threshold=tB,
                       labels=[i.label for i in structB.p1().sites],
                       leaf_rotation=-90)
            colB1.pyplot(figB)
        else:
            colB1.text('Primitive')
        netsB = densenets(structB, indices)
        dfB = pd.DataFrame()
        dfB['net'] = [
            [structB.p1().sites[j].label for j in i] for i in netsB['cluster']]
        dfB['width, A'] = netsB['width'][:]
        dfB['d_eff, 1/A^2'] = netsB['density'][:]
        colB1.dataframe(dfB)
        for i, s in enumerate(netsB['structure']):
            cifB += s.cif(f"B{i+1}") + '\n\n'

    st.download_button('Download CIF with separate nets in spread cell',
                       cifA+cifB,
                       file_name=f"{file.name[:-4]}_nets_"
                       f"{indices[0]}{indices[1]}{indices[2]}.cif",
                       use_container_width=True)
    P = spreadbasis(indices, structure.cell)
    st.download_button('Download CIF with selected sites in spread cell',
                       structure.sublatt(filtA+filtB).p1().transform(P).cif(),
                       file_name=f"{file.name[:-4]}_selected_"
                       f"{indices[0]}{indices[1]}{indices[2]}.cif",
                       use_container_width=True)
    st.download_button('Download CIF with all sites in spread cell',
                       structure.p1().transform(P).cif(),
                       file_name=f"{file.name[:-4]}_all_"
                       f"{indices[0]}{indices[1]}{indices[2]}.cif",
                       use_container_width=True)
