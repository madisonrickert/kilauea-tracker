"""UI layer for the Kīlauea Fountain Event Tracker.

Each submodule owns one piece of the surface. ``streamlit_app.py`` is a thin
orchestrator that composes these modules via ``st.tabs(...)``. Keeping view
code out of the Streamlit entrypoint makes the 2k-line monolith tractable and
testable.
"""
