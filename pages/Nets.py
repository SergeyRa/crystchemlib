from scipy.cluster.hierarchy import linkage, dendrogram
import core as ccl
import matplotlib.pyplot as plt
import nets as nt
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


file = st.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = st.selectbox('Choose datablock', parsed['datablock'])
structure = None
for d, s in zip(parsed['datablock'], parsed['structure']):
    if d == datablock:
        structure = s
if structure is None:
    labels = []
else:
    labels = [i.label for i in structure.sites]

st.divider()
col1, col2, col3 = st.columns(3)
h = col1.selectbox('h', range(-5, 6), 5)
k = col2.selectbox('k', range(-5, 6), 5)
l = col3.selectbox('l', range(-5, 6), 6)

if structure is not None:
    st.divider()
    colA, colB = st.columns(2)
    slA = colA.multiselect('Sublattice A', labels)
    filtA = structure.filter('label', slA)
    structA = structure.sublatt(filtA)
    ZA = structA.hklclust([h, k, l])
    if ZA is not None:
        figA = plt.figure()
        dendrogram(ZA,
                   #labels=[i.label for i in structA.p1().sites],
                   leaf_rotation=-90)
        colA.pyplot(figA)
    qqq = structA.med([h, k, l])
    qqq

    slB = colB.multiselect('Sublattice B', labels)
    filtB = structure.filter('label', slB)
    structB = structure.sublatt(filtB)
    projB = structB.hklproj([h, k, l])
    if len(projB) != 0:
        ZB = linkage(projB, method='single',
                     metric=lambda x, y:
                     nt.csd(x, y, ccl.dhkl(structB.cell, [h, k, l])))
        fig1 = plt.figure()
        dendrogram(ZB, labels=[i.label for i in structB.p1().sites],
                   leaf_rotation=-90)
        colB.pyplot(fig1)
