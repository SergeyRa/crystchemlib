import streamlit as st

st.text('Streamlit-based GUI for crystchemlib Python library\n'
        '(contact author at rashchenko@igm.nsc.ru '
        'for bug reports and suggestions)')
st.text('Known limitations and bugs in the current version:\n'
        '- Estimated standard deviations (esd) of bond lengths and angles\n'
        '  are calculated without symmetry constraints')
