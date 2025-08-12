import core as ccl
from io import StringIO
import pandas as pd
import streamlit as st


@st.cache_data
def parsefiles(files):
    parsed = {'source': [], 'data': [], 'loops': []}
    for file in files:
        fromfile = ccl.parsecif(file, None)
        for n, d, lo in zip(fromfile['name'],
                            fromfile['data'],
                            fromfile['loops']):
            if ccl.readstruct(d) is not None:
                parsed['source'].append(f'{file.name[:-4]}_{n}')
                parsed['data'].append(d)
                parsed['loops'].append(lo)
    return parsed


files = st.file_uploader("Choose CIF file(s)", accept_multiple_files=True)
parsed = parsefiles(files)
st.text(f'{len(parsed['data'])} datablock(s) with valid'
        f' structures were read from {len(files)} file(s)')

st.divider()
txt = st.text_area(
    'CIF keys to add (one per line followed by space-separated values;'
    f" first line should contain exactly {len(parsed['data'])} values)",
    height='content'
)
df1 = pd.DataFrame(parsed['source'], columns=['data_'])
if txt != '':
    df2 = pd.DataFrame()
    try:
        df2 = pd.read_csv(
            StringIO(txt), header=None, sep=' ', index_col=0, dtype=str
        ).transpose().reset_index(drop=True)
        df2.ffill(inplace=True)
        if len(df1.index) == len(df2.index):
            df1 = pd.concat([df1, df2], axis=1).fillna('?')
            for i in range(len(parsed['data'])):
                for j in df1.columns[1:]:
                    if j in parsed['data'][i].keys():
                        parsed['data'][i][j] = str(df1[j].iloc[i])
                    else:
                        parsed['data'][i][j] = str(df1[j].iloc[i])
                        parsed['data'][i].move_to_end(j, last=False)
        else:
            st.text('Wrong format of new keys!')
    except pd.errors.ParserError:
        st.text('Wrong format of new keys!')
st.dataframe(df1, hide_index=True)

merged = ''
for s, d, lo in zip(parsed['source'],
                    parsed['data'],
                    parsed['loops']):
    merged += ccl.printcif(d, lo, name=s)
    merged += '\n'
st.download_button('Download merged CIF', merged, file_name='merged.cif',
                   type='primary', use_container_width=True)
