import core as ccl
import pandas as pd
import streamlit as st


@st.cache_data
def parsefiles(files):
    parsed = {'source': [], 'data': []}
    for file in files:
        fromfile = ccl.parsecif(file, None, True)
        for n, d in zip(fromfile['name'], fromfile['data']):
            d = ccl.clearkeys(d)[0]
            parsed['source'].append(f'{file.name} ({n})')
            parsed['data'].append(d)
    return parsed


files = st.file_uploader("Choose CIF file(s)", accept_multiple_files=True)
parsed = parsefiles(files)
st.text(f"{len(parsed['data'])} datablock(s) "
        f"were read from {len(files)} file(s)")

allkeys = set([])
for i in parsed['data']:
    if i is not None:
        allkeys = allkeys.union(set(i.keys()))

st.divider()
col1, col2 = st.columns(2)
num = col1.multiselect("Choose keys for numeric import", sorted(allkeys))
txt = col2.multiselect("Choose keys for text import", sorted(allkeys))

df = {'source': []}
for i in num:
    df[i] = []
    df[i+'_esd'] = []
for i in txt:
    df[i] = []
for s, d in zip(parsed['source'], parsed['data']):
    df['source'].append(s)
    for i in num:
        if i in d:
            val, esd = ccl.readesd(d[i])
        else:
            val, esd = (None, None)
        df[i].append(val)
        df[i+'_esd'].append(esd)
    for i in txt:
        if i in d:
            val = d[i]
        else:
            val = None
        df[i].append(val)
df = pd.DataFrame(df)

if len(df['source']) != 0:
    st.divider()
    df
