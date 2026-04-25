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

/* In-content page-link CTAs (the Now page's "View full prediction model"
   and "View all cameras" jumps). Streamlit's default st.page_link is a
   small underlined-text link; we re-skin it as a centered lava pill so it
   reads as a primary CTA against the volcano palette. The selector targets
   stPageLink (in-content) — NOT stPageLink-NavLink (the nav-bar variant),
   which is styled by Streamlit's header CSS. */
[data-testid="stPageLink"] {{
    margin: var(--space-md) 0 var(--space-lg);
    text-align: center;
}}
[data-testid="stPageLink"] a {{
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
[data-testid="stPageLink"] a:hover,
[data-testid="stPageLink"] a:focus {{
    background: rgba(255, 107, 53, 0.18);
    border-color: var(--lava);
    color: var(--lava) !important;
}}
[data-testid="stPageLink"] a span,
[data-testid="stPageLink"] a p {{
    color: inherit;
    font-weight: inherit;
    font-size: inherit;
    margin: 0;
}}

/* Compact camera strip on the Now tab. */
.kt-cam-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-top: var(--space-lg);
}}

/* Primary navigation — Streamlit's top-positioned st.navigation renders
   inside the header bar as ``[data-testid="stTopNavLink"]``. The defaults
   look fine; this block bumps the font weight and active-state contrast
   so the current page reads clearly against the obsidian background. */
[data-testid="stTopNavLink"] a {{
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--steam);
    opacity: 0.7;
    transition: opacity 120ms ease;
}}
[data-testid="stTopNavLink"] a:hover {{
    opacity: 0.95;
}}
[data-testid="stTopNavLink"] a[aria-current="page"] {{
    opacity: 1;
    color: var(--lava);
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

/* Loading spinner / status overlay — viewport-centered, volcano-tinted.
   Two cache-miss surfaces fire here: ``st.cache_data(show_spinner="…")``
   for the fast tilt-history CSV read, and an ``st.status`` widget for the
   slow ``cached_ingest`` path that walks five USGS sources and reconciles.
   Both render at module scope before any content is on screen. Streamlit's
   default placement honors the wide-layout column (which makes the inline
   placeholder land middle-LEFT, not viewport-center), so we promote the
   element to a fixed full-viewport overlay with a dim+blur backdrop. The
   icon keeps its lava drop-shadow and gets a slow pulsing ring for a touch
   of life without crossing into ``cute``. */
/* The running st.status widget is rendered as an expander whose summary
   contains an ``<i data-testid="stExpanderIconSpinner">`` Material Symbols
   icon. That icon only exists while the status is in the "running" state —
   Streamlit swaps it for a check / cross when the state flips to complete
   or error. The ``:has()`` guard means our overlay only covers the viewport
   while ingest is actually working, and stops matching the moment work
   finishes — regular expanders elsewhere in the app are left untouched.
   The toolbar running indicator (``stStatusWidget``) is a different element
   and we deliberately do NOT target it here. */
[data-testid="stSpinner"],
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) {{
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-lg);
    padding: var(--space-xl) var(--space-lg);
    text-align: center;
    background: rgba(15, 20, 25, 0.92);
    -webkit-backdrop-filter: blur(6px);
    backdrop-filter: blur(6px);
}}
/* Inner <details> stretches to full width by default; force it to shrink
   to its content so the summary lands in the true horizontal center. */
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details {{
    width: auto;
    max-width: 90vw;
    border: none;
    background: transparent;
    box-shadow: none;
}}
/* The visible row: spinner icon + status label. Render as an inline flex
   so the icon and label sit side-by-side and the whole thing is sized to
   content (centerable by the parent overlay's flex layout). */
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details > summary {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-md);
    padding: var(--space-sm) var(--space-md);
    background: transparent;
    border: none;
    list-style: none;
    cursor: default;
}}
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details > summary::-webkit-details-marker {{ display: none; }}
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details > summary::marker {{ content: ""; }}
/* Hide the collapsed body region — when expanded=False there's nothing
   useful in it and we don't want it adding stray whitespace below the
   centered summary. */
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) [data-testid="stExpanderDetails"] {{
    display: none !important;
}}
[data-testid="stSpinner"] > div,
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) [data-testid="stMarkdownContainer"] p {{
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--steam);
    letter-spacing: 0.01em;
    margin: 0;
}}
@keyframes kt-spin {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}
@keyframes kt-spinner-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.55); }}
    50%      {{ box-shadow: 0 0 0 18px rgba(255, 107, 53, 0); }}
}}
/* Spin + pulse compose because they target separate properties (transform
   vs box-shadow). The bg/border/padding resets clear whatever the
   underlying Streamlit emotion classes were drawing on the <i> — that
   was the rectangular "bounding box" visible behind a non-spinning glyph
   in the broken-state screenshot. ``display: inline-block`` so the pulse
   box-shadow renders around the icon's full box rather than its line
   box (an inline ``<i>`` would clip the shadow at the baseline). */
[data-testid="stSpinner"] svg,
[data-testid="stSpinner"] i,
[data-testid="stExpanderIconSpinner"] {{
    color: var(--lava) !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    border-radius: 50%;
    font-size: 2.25rem !important;
    display: inline-block;
    filter: drop-shadow(0 0 14px rgba(255, 107, 53, 0.55));
    animation: kt-spin 1s linear infinite,
               kt-spinner-pulse 2s ease-out infinite;
}}
/* Streamlit's spinner row and the expander's <details>/<summary> default
   to ``overflow: hidden`` (and the summary clips at the line box), which
   chops the outer edge off the pulsing box-shadow. Force ``overflow:
   visible`` all the way down so the 18px ring renders fully. The icon
   inside the expander summary is wrapped in an extra emotion-styled
   ``<span>`` — that's the actual clipping ancestor in the rendered DOM,
   so ``:has()`` it explicitly. */
[data-testid="stSpinner"],
[data-testid="stSpinner"] > div,
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details,
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) > details > summary,
[data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) span:has([data-testid="stExpanderIconSpinner"]) {{
    overflow: visible !important;
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
    .kt-topbar {{ flex-direction: column; align-items: stretch; }}
    [data-testid="stSpinner"] > div,
    [data-testid="stExpander"]:has([data-testid="stExpanderIconSpinner"]) [data-testid="stMarkdownContainer"] p {{ font-size: 1.25rem; }}
}}
</style>"""
