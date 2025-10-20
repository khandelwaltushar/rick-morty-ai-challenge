import os
from typing import Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


SYSTEM_SUMMARY = (
    "You are a witty, sardonic Rick & Morty narrator who keeps facts accurate."
)


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _openai_chat(client, prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_SUMMARY},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def summarize_location(location: Dict[str, any]) -> str:
    name = location.get("name", "Unknown Location")
    type_ = location.get("type", "Unknown Type")
    dimension = location.get("dimension", "Unknown Dimension")
    residents = location.get("residents", [])
    resident_names = ", ".join([r.get("name", "?") for r in residents[:8]])

    prompt = (
        f"Summarize the location '{name}' in 4-6 sentences in the style of a Rick & Morty narrator. "
        f"Keep facts accurate. Mention its type ('{type_}'), dimension ('{dimension}'), and give a sense of notable residents (e.g., {resident_names})."
    )

    client = _get_openai_client()
    if client:
        try:
            return _openai_chat(client, prompt)
        except Exception:
            pass

    # Fallback template-based generation
    lines = [
        f"Welcome to {name}, a {type_.lower()} tucked inside the {dimension}.",
        "It's exactly the kind of place where portal mistakes feel intentional.",
        f"Locals range from the polite to the gelatinous: {resident_names or 'the usual suspects'}.",
        "If you hear belching, that's just the ambiance. If you hear screaming, that's also the ambiance.",
        "Anyway, watch your step. Gravity is more of a suggestion around here.",
    ]
    return " " .join(lines)


def generate_dialogue(ch1: Dict[str, any], ch2: Dict[str, any]) -> str:
    n1 = ch1.get("name", "Character A")
    n2 = ch2.get("name", "Character B")
    s1 = ch1.get("species", "?")
    s2 = ch2.get("species", "?")
    prompt = (
        f"Write a playful 6-8 line dialogue between {n1} (species: {s1}) and {n2} (species: {s2}). "
        "Keep it witty and on-brand for Rick & Morty, without violating safety policies."
    )

    client = _get_openai_client()
    if client:
        try:
            return _openai_chat(client, prompt)
        except Exception:
            pass

    # Fallback scripted banter
    return (
        f"{n1}: You ever feel like someone's binge-watching our lives?\n"
        f"{n2}: Only when I'm interesting. So, never.\n"
        f"{n1}: I'm {s1}, interesting is my species trait.\n"
        f"{n2}: I'm {s2}. My trait is surviving monologues.\n"
        f"{n1}: Wanna grab a portal and regret it?\n"
        f"{n2}: Regret is my cardio. Let's go."
    )
