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

/* Hero caption — one-line supporting caption rendered BELOW the sparkline.
   Replaces the old big-headline hero-card once the sparkline was promoted
   to be the dominant visual. Chip + eyebrow + headline + subhead flow on
   a single wrapping flex row so wide screens read as one line and mobile
   stacks naturally. */
.kt-hero-caption {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-md);
    padding: var(--space-md) var(--space-lg);
    background: var(--basalt);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    margin-top: calc(var(--space-md) * -1);  /* tuck it close under the sparkline */
    margin-bottom: var(--space-lg);
}}
.kt-hero-caption__eyebrow {{
    color: var(--steam);
    opacity: 0.55;
    font-size: 0.875rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.kt-hero-caption__headline {{
    color: var(--steam);
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.01em;
}}
.kt-hero-caption__subhead {{
    color: var(--steam);
    opacity: 0.75;
    font-family: var(--font-mono);
    font-size: 1rem;
}}

/* Legacy hero block — one dramatic answer, not three metrics.
   Retained for ``render_html`` (backward-compat with tests and any
   caller that still builds the five-element stacked card). The live
   app uses ``render_caption_html`` + sparkline above it. */
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

/* Floating inspector-overlay dock — on desktop the "Inspector overlay
   layers" expander docks to the top-right corner as a fixed panel so the
   user can toggle layers while scrolling through PNG previews below. On
   mobile (< 1100px) it stays in-flow above the inspector. The
   ``.kt-overlay-dock`` class is added at runtime by a MutationObserver
   (see streamlit_app.py) because st.container doesn't accept a class. */
@media (min-width: 1100px) {{
    .kt-overlay-dock {{
        position: fixed;
        top: 110px;
        right: 16px;
        width: 300px;
        max-height: calc(100vh - 140px);
        overflow-y: auto;
        z-index: 80;
        background: var(--basalt);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid var(--lava);
        border-radius: 10px;
        padding: 4px var(--space-sm);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }}
    .kt-overlay-dock [data-testid="stExpander"] {{
        border: none;
        background: transparent;
    }}
}}

/* Teaching-tool diagnostic chips — replace plain st.metric displays
   inside the Pipeline tab's Model diagnostics expander. Each chip shows
   label + value (tinted by classification) + a colored verdict badge +
   a one-line expected-range note. The verdict color is driven inline via
   the chip's HTML style attribute using palette tokens. */
.kt-diag-chip {{
    background: var(--basalt);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: var(--space-md);
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-height: 140px;
}}
.kt-diag-chip__label {{
    color: var(--steam);
    opacity: 0.65;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.kt-diag-chip__value {{
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.05;
    margin-top: 2px;
}}
.kt-diag-chip__unit {{
    font-size: 0.875rem;
    font-weight: 400;
    opacity: 0.6;
    margin-left: 4px;
}}
.kt-diag-chip__verdict {{
    display: inline-block;
    align-self: flex-start;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}}
.kt-diag-chip__note {{
    color: var(--steam);
    opacity: 0.65;
    font-size: 0.8125rem;
    line-height: 1.4;
    margin-top: 4px;
}}

/* CTA row — button-styled links in the Now tab that jump to other tabs
   via the JS tab router (``?tab=chart`` etc). Renders as a filled lava
   button with obsidian text — easier to read than an outlined variant
   and high-specificity enough to beat Streamlit's default link color
   (which would otherwise render the text blue). */
.kt-cta-row {{
    margin: var(--space-md) 0 var(--space-lg);
    text-align: center;
}}
a.kt-cta,
a.kt-cta:link,
a.kt-cta:visited,
[data-testid="stMarkdownContainer"] a.kt-cta {{
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    padding: 9px 18px;
    background: rgba(255, 107, 53, 0.08);
    border: 1px solid rgba(255, 107, 53, 0.35);
    border-radius: 8px;
    color: var(--lava) !important;
    font-weight: 600;
    font-size: 0.9375rem;
    letter-spacing: 0.01em;
    text-decoration: none !important;
    transition: background 140ms ease, border-color 140ms ease;
}}
a.kt-cta:hover,
a.kt-cta:focus,
[data-testid="stMarkdownContainer"] a.kt-cta:hover,
[data-testid="stMarkdownContainer"] a.kt-cta:focus {{
    background: rgba(255, 107, 53, 0.18);
    border-color: var(--lava);
    color: var(--lava) !important;
}}

/* Compact camera strip on the Now tab. */
.kt-cam-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-top: var(--space-lg);
}}

/* Primary navigation — Streamlit tabs upgraded to look like a nav bar.
   Streamlit's default tab labels are small and the active underline is
   narrow, so first-time visitors didn't read them as primary nav. */
[data-testid="stTabs"] > div > div[role="tablist"] {{
    gap: var(--space-lg);
    border-bottom: 1px solid rgba(226, 232, 240, 0.08);
    padding: 0 var(--space-xs);
    margin-bottom: var(--space-lg);
}}
[data-testid="stTabs"] button[role="tab"] {{
    font-size: 1.0625rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: var(--space-md) var(--space-sm);
    color: var(--steam);
    opacity: 0.6;
    transition: opacity 120ms ease, color 120ms ease;
    border-bottom: 3px solid transparent;
    margin-bottom: -1px;  /* overlap the tablist's bottom border */
}}
[data-testid="stTabs"] button[role="tab"]:hover {{
    opacity: 0.85;
    color: var(--steam);
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    opacity: 1;
    color: var(--steam);
    border-bottom-color: var(--lava);
}}

/* Top bar — compact always-visible header strip above the tabs.
   Holds refresh + tz + freshness caption. Replaces the full sidebar. */
.kt-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-md);
    padding: var(--space-sm) 0;
    margin-bottom: var(--space-xs);
    flex-wrap: wrap;
}}
.kt-topbar__meta {{
    color: var(--steam);
    opacity: 0.65;
    font-size: 0.8125rem;
    font-family: var(--font-mono);
    display: flex;
    flex-direction: column;
    gap: 2px;
    line-height: 1.4;
    text-align: right;
}}

/* Footer — sticky attribution strip pinned to the bottom of the viewport.
   Uses a deeper basalt background + top border so it reads as distinct
   chrome from the page content. A GitHub icon sits inline with the
   "source on GitHub" link so the footer carries a recognizable visual
   anchor.

   Streamlit's main container adds its own bottom padding, but a sticky
   footer can still overlap long-scrolling content. The extra body
   bottom-padding (.kt-footer-spacer) reserves space so the last
   in-page element isn't hidden behind the footer. */
.kt-footer {{
    position: sticky;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 100;
    margin: var(--space-xl) calc(var(--space-lg) * -1) 0;
    padding: var(--space-sm) var(--space-lg);
    background: var(--basalt);
    border-top: 1px solid rgba(226, 232, 240, 0.10);
    color: var(--steam);
    opacity: 0.95;
    font-size: 0.8125rem;
    text-align: center;
    backdrop-filter: blur(6px);
}}
.kt-footer__inner {{
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    flex-wrap: wrap;
    justify-content: center;
}}
.kt-footer a {{
    color: var(--lava);
    text-decoration: none;
    opacity: 0.9;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}
.kt-footer a:hover {{
    opacity: 1;
    text-decoration: underline;
}}
.kt-footer svg {{
    width: 14px;
    height: 14px;
    vertical-align: -2px;
    fill: currentColor;
}}
.kt-footer__sep {{
    opacity: 0.4;
}}

/* Hero sparkline — sits inside the kt-hero card, below the headline.
   Meant to read as a visual fingerprint of recent activity, not a
   readable chart — no axes, no legend, no hover. */
.kt-spark {{
    margin-top: var(--space-md);
    border-top: 1px solid rgba(226, 232, 240, 0.06);
    padding-top: var(--space-md);
}}
.kt-spark__label {{
    color: var(--steam);
    opacity: 0.55;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: var(--space-xs);
}}

/* Loading spinner — big, centered, volcano-tinted.
   Streamlit's default ``st.cache_data(show_spinner="…")`` renders a small
   inline row that reads like a micro toast. Both spinners in this app fire
   at module scope (before any content is on screen), so we let the spinner
   own the viewport while the cache misses: large headline text, vertically
   centered via ``min-height: 60vh``, with a lava-tinted glow on the native
   spinner icon so it feels on-theme. */
[data-testid="stSpinner"] {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    padding: var(--space-xl) var(--space-lg);
    gap: var(--space-lg);
    text-align: center;
}}
[data-testid="stSpinner"] > div {{
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--steam);
    letter-spacing: 0.01em;
    gap: var(--space-md);
}}
/* Streamlit's own spinner icon — bump its size and tint the glow. */
[data-testid="stSpinner"] svg {{
    width: 44px !important;
    height: 44px !important;
    filter: drop-shadow(0 0 14px rgba(255, 107, 53, 0.55));
}}
[data-testid="stSpinner"] i {{
    font-size: 2.25rem !important;
    filter: drop-shadow(0 0 14px rgba(255, 107, 53, 0.55));
}}

/* Breakpoints — explicit, not Streamlit defaults. */
@media (max-width: 1024px) {{
    .kt-cam-strip {{ grid-template-columns: repeat(2, 1fr); }}
    .kt-hero__headline {{ font-size: 4rem; }}
    [data-testid="stTabs"] > div > div[role="tablist"] {{ gap: var(--space-md); }}
}}
@media (max-width: 640px) {{
    .kt-cam-strip {{ grid-template-columns: 1fr; }}
    .kt-hero {{ padding: var(--space-md); }}
    .kt-hero__headline {{ font-size: 3rem; }}
    [data-testid="stTabs"] button[role="tab"] {{ font-size: 0.9375rem; }}
    .kt-topbar {{ flex-direction: column; align-items: stretch; }}
    [data-testid="stSpinner"] > div {{ font-size: 1.25rem; }}
}}
</style>"""
