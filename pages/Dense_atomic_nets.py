from nets import Structure
import core as ccl
import pandas as pd
import streamlit as st


def hash_func(obj):
    return obj.cif()


@st.cache_data(hash_funcs={Structure: hash_func})
def densenets(structure, indices):
    return structure.densenets(indices, fig=True)


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
    df = structure.searchnets(groups, hklmax=[maxhkl]*3)
    df.rename(columns={'mean': 'mean d_eff, 1/A^2',
                       'equiv': 'equivalent hkl for Laue class'},
              inplace=True)
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
    cif = ""
    structAB = structure.sublatt(filtA+filtB)
    netsAB = densenets(structAB, indices)
    df = pd.DataFrame()
    df['net'] = [
        [structAB.p1().sites[j].label for j in i] for i in netsAB['cluster']]
    df['d_eff, 1/A^2'] = netsAB['density']
    df['width, A'] = netsAB['width']
    df['z'] = netsAB['z']
    st.dataframe(df, hide_index=True)
    if netsAB['fig'] is not None:
        st.pyplot(netsAB['fig'])
    for i, s in enumerate(netsAB['structure']):
        cif += s.cif(f"{i+1}") + '\n\n'

    st.download_button('Download CIF with nets in spread cell',
                       cif,
                       file_name=f"{file.name[:-4]}_nets_"
                       f"{indices[0]}{indices[1]}{indices[2]}.cif",
                       use_container_width=True)
