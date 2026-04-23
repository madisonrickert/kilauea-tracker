"""Global CSS for the app — palette tokens + typography + spacing + breakpoints.

Emits a single ``<style>`` block that ``streamlit_app.py`` injects once near
page boot. Keeping the CSS in a Python string (rather than a separate .css
file) sidesteps Streamlit's static-asset story and lets tests assert token
completeness directly.
"""

from __future__ import annotations

from .palette import ALL_TOKENS

# Typography
_INTER_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;600;800&display=swap');"
)
_FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, Oxygen-Sans, Ubuntu, sans-serif"
)
_MONO_STACK = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# Spacing scale — 4 / 8 / 16 / 24 / 48 px
_SPACE = {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "48px"}


def _css_variables() -> str:
    """Emit ``--token: value;`` lines for every palette + spacing + type token."""
    lines: list[str] = []
    for name, value in ALL_TOKENS.items():
        lines.append(f"    --{name}: {value};")
    for key, value in _SPACE.items():
        lines.append(f"    --space-{key}: {value};")
    lines.append(f"    --font-body: {_FONT_STACK};")
    lines.append(f"    --font-mono: {_MONO_STACK};")
    return "\n".join(lines)


def build_style_block() -> str:
    """Return the full ``<style>…</style>`` block to inject at app boot."""
    return f"""<style>
{_INTER_IMPORT}

:root {{
{_css_variables()}
}}

html, body, [class*="css"] {{
    font-family: var(--font-body);
}}

/* Preserve Streamlit's native metric value layout but swap the family. */
[data-testid="stMetricValue"] {{
    white-space: normal !important;
    overflow-wrap: anywhere;
    line-height: 1.1;
    font-size: 1.6rem;
    font-family: var(--font-body);
    font-weight: 600;
}}

/* Hero block — one dramatic answer, not three metrics. */
.kt-hero {{
    background: var(--basalt);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: var(--space-lg) var(--space-xl);
    margin-bottom: var(--space-lg);
}}
.kt-hero__eyebrow {{
    color: var(--steam);
    opacity: 0.65;
    font-size: 0.875rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: var(--space-sm);
}}
.kt-hero__headline {{
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    color: var(--steam);
    margin: 0;
}}
.kt-hero__subhead {{
    font-family: var(--font-mono);
    font-size: 1.125rem;
    color: var(--steam);
    opacity: 0.75;
    margin-top: var(--space-sm);
}}
.kt-hero__explainer {{
    color: var(--steam);
    opacity: 0.85;
    margin-top: var(--space-md);
    font-size: 1rem;
    line-height: 1.5;
}}

/* State chip — consistent shape; color is set per-state via --chip-bg inline. */
.kt-chip {{
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--chip-bg, var(--ash));
    color: var(--steam);
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

/* Structured state banner — three-part. */
.kt-banner {{
    background: var(--basalt);
    border-left: 4px solid var(--banner-accent, var(--ash));
    padding: var(--space-md) var(--space-lg);
    border-radius: 8px;
    margin: var(--space-md) 0;
}}
.kt-banner__headline {{
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--steam);
    margin: 0 0 var(--space-sm) 0;
}}
.kt-banner__explanation {{
    color: var(--steam);
    opacity: 0.85;
    margin: 0;
    line-height: 1.5;
}}

/* Focus states for keyboard users. */
*:focus-visible {{
    outline: 2px solid var(--lava);
    outline-offset: 2px;
    border-radius: 2px;
}}

/* Compact camera strip on the Now tab. */
.kt-cam-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-top: var(--space-lg);
}}

/* Breakpoints — explicit, not Streamlit defaults. */
@media (max-width: 1024px) {{
    .kt-cam-strip {{ grid-template-columns: repeat(2, 1fr); }}
    .kt-hero__headline {{ font-size: 4rem; }}
}}
@media (max-width: 640px) {{
    .kt-cam-strip {{ grid-template-columns: 1fr; }}
    .kt-hero {{ padding: var(--space-md); }}
    .kt-hero__headline {{ font-size: 3rem; }}
}}
</style>"""
