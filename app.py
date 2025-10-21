import os
from typing import Dict, List, Optional

import streamlit as st

from data_client import get_locations, get_characters_index
from db import init_db, add_note, list_notes_by_character, list_all_notes
from gen import summarize_location, generate_dialogue
from eval import evaluate_generation
from embeddings import cosine_rank


st.set_page_config(page_title="Rick & Morty AI", layout="wide")

# Initialize DB
init_db()

@st.cache_data(show_spinner=False)
def load_locations() -> List[Dict[str, any]]:
    return get_locations()


def sidebar_locations(locations: List[Dict[str, any]]) -> Optional[Dict[str, any]]:
    st.sidebar.header("Locations")
    loc_options = {f"{loc['name']} ({loc['type']})": loc for loc in locations}
    selected = st.sidebar.selectbox("Choose a location", list(loc_options.keys()))
    return loc_options[selected]


def render_character_card(ch: Dict[str, any]):
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(ch.get("image"), width=120)
    with cols[1]:
        st.subheader(ch.get("name"))
        st.caption(f"{ch.get('species')} • {ch.get('status')} • {ch.get('gender')}")
        st.caption(f"Origin: {ch.get('origin', {}).get('name', 'Unknown')}")


def render_notes_section(character_id: str, character_name: str):
    st.markdown("---")
    st.markdown("**Notes**")
    existing = list_notes_by_character(character_id)
    for n in existing:
        st.write(f"- {n['content']}  ")
        st.caption(n["created_at"]) 
    note = st.text_area("Add a note", key=f"note_{character_id}")
    if st.button("Save note", key=f"save_{character_id}") and note.strip():
        add_note(character_id, character_name, note.strip())
        st.rerun()


def main():
    st.title("Rick & Morty: Retrieval + Generation + Evaluation")

    locations = load_locations()
    selected_loc = sidebar_locations(locations)

    st.markdown(f"### Location: {selected_loc.get('name')}  ")
    st.caption(f"Type: {selected_loc.get('type')} • Dimension: {selected_loc.get('dimension')}")

    residents = selected_loc.get("residents", [])

    with st.expander("Residents", expanded=True):
        for ch in residents:
            with st.container(border=True):
                render_character_card(ch)
                render_notes_section(ch.get("id"), ch.get("name"))

    st.markdown("---")
    st.subheader("Generative features")

    tab1, tab2, tab3 = st.tabs(["Summarize location", "Dialogue", "Search"]) 

    with tab1:
        st.write("Generate a narrator-style summary and evaluate it.")
        if st.button("Generate summary"):
            with st.spinner("Generating..."):
                summary = summarize_location(selected_loc)
            st.write(summary)
            scores = evaluate_generation(summary, selected_loc)
            st.write({k: round(v, 3) for k, v in scores.items()})

    with tab2:
        ch_idx = get_characters_index()
        character_names = [c.get("name") for c in ch_idx.values()]
        colA, colB = st.columns(2)
        with colA:
            pick1 = st.selectbox("Character A", character_names, key="dlg_a")
        with colB:
            pick2 = st.selectbox("Character B", character_names, key="dlg_b")
        if st.button("Generate dialogue"):
            ch1 = next((c for c in ch_idx.values() if c.get("name") == pick1), None)
            ch2 = next((c for c in ch_idx.values() if c.get("name") == pick2), None)
            if ch1 and ch2:
                with st.spinner("Generating..."):
                    dlg = generate_dialogue(ch1, ch2)
                st.text(dlg)
            else:
                st.warning("Pick two valid characters.")

    with tab3:
        st.write("AI-augmented semantic search across notes and residents (embeddings + cosine).")
        q = st.text_input("Search query")
        if q:
            corpus_rows = list_all_notes()
            # Include resident names as pseudo-notes for better coverage
            for ch in residents:
                corpus_rows.append({
                    "id": int(ch.get("id")),
                    "character_id": ch.get("id"),
                    "character_name": ch.get("name"),
                    "content": f"[resident] {ch.get('name')} {ch.get('species')} {ch.get('status')} {selected_loc.get('name')} {selected_loc.get('type')}",
                    "created_at": "",
                })
            corpus_texts = [r["content"] for r in corpus_rows]
            ranked = cosine_rank(q, corpus_texts, top_k=10)
            for idx, sim in ranked:
                row = corpus_rows[idx]
                st.write(f"{row['character_name']}: {row['content']}")
                st.caption(f"score={sim:.3f}")


if __name__ == "__main__":
    main()
