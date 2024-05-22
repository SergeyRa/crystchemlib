from scipy.cluster.hierarchy import dendrogram
from nets import Structure, spreadbasis
import core as ccl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


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


def hash_func(obj):
    return obj.cif()


@st.cache_data(hash_funcs={Structure: hash_func})
def searchnets(structure, groups):
    result = structure.searchnets(groups)
    df = pd.DataFrame()
    df['hkl'] = result['hkl'][:]
    df['mean eff. density, 1/A^2'] = result['mean'][:]
    df['max eff. density, 1/A^2'] = result['max'][:]
    df['equivalent hkl for Laue class'] = [
        ccl.equivhkl(structure.symops, i) for i in result['hkl']]
    return df


col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col2.selectbox('Choose datablock', parsed['datablock'])

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

if (slA != []) or (slB != []):
    filtA = structure.filter('label', slA)
    filtB = structure.filter('label', slB)
    result = searchnets(structure, [filtA, filtB])
    st.divider()
    st.dataframe(result)
    indices = st.selectbox('Choose hkl from table', result['hkl'])

    st.divider()
    colA1, colB1 = st.columns(2)

    colA1.text('Sublattice A')
    cifA = ""
    if slA != []:
        structA = structure.sublatt(filtA)
        ZA = structA.hklclust(indices)
        if len(ZA) > 0:
            figA = plt.figure()
            dendrogram(ZA,
                       labels=[i.label for i in structA.p1().sites],
                       leaf_rotation=-90)
            colA1.pyplot(figA)
        else:
            colA1.text('Primitive')
        netsA = structA.densenets(indices)
        dfA = pd.DataFrame()
        dfA['net'] = [
            [structA.p1().sites[j].label for j in i] for i in netsA['cluster']]
        dfA['width, A'] = netsA['width'][:]
        dfA['eff. density, 1/A^2'] = netsA['density'][:]
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
            dendrogram(ZB,
                       labels=[i.label for i in structB.p1().sites],
                       leaf_rotation=-90)
            colB1.pyplot(figB)
        else:
            colB1.text('Primitive')
        netsB = structB.densenets(indices)
        dfB = pd.DataFrame()
        dfB['net'] = [
            [structB.p1().sites[j].label for j in i] for i in netsB['cluster']]
        dfB['width, A'] = netsB['width'][:]
        dfB['eff. density, 1/A^2'] = netsB['density'][:]
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
                       file_name=f"{file.name[:-4]}_"
                       f"{indices[0]}{indices[1]}{indices[2]}.cif",
                       use_container_width=True)
