"""
Show Me the Orbit — Dash web app
Sarah Blunt (2019), web port (2026)

Run: python app.py  →  open http://localhost:8050
"""

from functools import lru_cache

import numpy as np
import astropy.units as u
import astropy.constants as consts
import orbitize.kepler

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# ── Fixed orbital parameters ────────────────
SMA = 10  # semi-major axis [au]
TAU = 0.3  # epoch of periastron [fraction of period past MJD=0]
PLX = 30  # parallax [mas]
MTOT = 1.3  # total mass [Msun]
MPLANET = 0.01  # planet mass [Msun]
TIME_DIVS = 500  # points along the orbit
MAX_ECC = 0.95  # slider upper limit

per_yr = np.sqrt(SMA**3 / MTOT)
per_days = per_yr * 365.25
max_sep = SMA * (1 + MAX_ECC) * PLX  # largest separation [mas]

maxK = (
    (
        np.sqrt(consts.G / (1.0 - MAX_ECC**2))
        * (MPLANET * u.Msun)
        / np.sqrt(MTOT * u.Msun)
        / np.sqrt(SMA * u.au)
    )
    .to(u.km / u.s)
    .value
)

epochs = np.linspace(0, per_days, TIME_DIVS)
epochs_yr = epochs / 365.25

# Trace indices — orbit figure
IO_BACK = 0  # faded arc (behind sky plane)
IO_FRONT = 1  # solid arc (in front of sky plane)
IO_NODES = 2  # line of nodes (dashed)
IO_STAR = 3  # star marker
IO_PLANET = 4  # animated planet marker

# Trace indices — RV figure
IR_TRACK = 0  # RV curve
IR_ZERO = 1  # zero line
IR_PLANET = 2  # animated planet marker


# ── Orbital computation ──────────────────────────────────────────────────────


@lru_cache(maxsize=32)
def _base_orbit(ecc):
    """Base orbit (inc=aop=pan=0) for z-depth calculation. Cached by ecc only."""
    ra0, dec0, _ = orbitize.kepler.calc_orbit(
        epochs, SMA, ecc, 0, 0, 0, TAU, PLX, MTOT, mass_for_Kamp=MPLANET
    )
    return ra0, dec0


@lru_cache(maxsize=256)
def compute_orbit(ecc, inc_deg, aop_deg, pan_deg):
    """Return ra, dec, rv, z arrays using orbitize."""
    inc = np.radians(inc_deg)
    aop = np.radians(aop_deg)
    pan = np.radians(pan_deg)

    ra, dec, rv = orbitize.kepler.calc_orbit(
        epochs,
        SMA,
        ecc,
        inc,
        aop,
        pan,
        TAU,
        PLX,
        MTOT,
        mass_for_Kamp=MPLANET,
    )
    rv = (
        -rv
    )  # stellar RVs must be multiplied by -1 because we're using aop of the planet here.

    ra0, dec0 = _base_orbit(ecc)
    z = np.sin(inc) * (-np.cos(aop) * ra0 - np.sin(aop) * dec0) * PLX

    return ra.tolist(), dec.tolist(), rv.tolist(), z.tolist()


# ── Figure builders ──────────────────────────────────────────────────────────

PURPLE = "rgba(128,0,128,1.0)"
PURPLE_FADED = "rgba(128,0,128,0.12)"


def build_orbit_figure(ra, dec, z, pan_deg, idx):
    pan = np.radians(pan_deg)

    # Split orbit track into front (z≥0) and back (z<0) segments; None = line break
    front_ra = [r if z[i] >= 0 else None for i, r in enumerate(ra)]
    front_dec = [d if z[i] >= 0 else None for i, d in enumerate(dec)]
    back_ra = [r if z[i] < 0 else None for i, r in enumerate(ra)]
    back_dec = [d if z[i] < 0 else None for i, d in enumerate(dec)]

    # Line of nodes direction given by PAN
    lon_x = [-np.sin(pan) * max_sep, np.sin(pan) * max_sep]
    lon_y = [-np.cos(pan) * max_sep, np.cos(pan) * max_sep]

    planet_opacity = 1.0 if z[idx] >= 0 else 0.15

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(  # IO_BACK
            x=back_ra,
            y=back_dec,
            mode="lines",
            line=dict(color=PURPLE_FADED, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IO_FRONT
            x=front_ra,
            y=front_dec,
            mode="lines",
            line=dict(color=PURPLE, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IO_NODES
            x=lon_x,
            y=lon_y,
            mode="lines",
            line=dict(color="#555555", dash="dash", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IO_STAR
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                symbol="star",
                size=18,
                color="violet",
                line=dict(color="black", width=1),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IO_PLANET
            x=[ra[idx]],
            y=[dec[idx]],
            mode="markers",
            marker=dict(
                size=10,
                color="violet",
                opacity=planet_opacity,
                line=dict(color="black", width=1),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#333333", size=13, family="system-ui, sans-serif"),
        margin=dict(t=20, b=55, l=70, r=20),
        xaxis=dict(
            title="ΔRA [mas]",
            range=[1.1 * max_sep, -1.1 * max_sep],  # RA increases East (left)
            gridcolor="#e8e8e8",
            zerolinecolor="#bbbbbb",
        ),
        yaxis=dict(
            title="ΔDec [mas]",
            range=[-1.1 * max_sep, 1.1 * max_sep],
            gridcolor="#e8e8e8",
            zerolinecolor="#bbbbbb",
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return fig


def build_rv_figure(rv, idx):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(  # IR_TRACK
            x=epochs_yr.tolist(),
            y=rv,
            mode="lines",
            line=dict(color="purple", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IR_ZERO
            x=[float(epochs_yr[0]), float(epochs_yr[-1])],
            y=[0, 0],
            mode="lines",
            line=dict(color="#aaaaaa", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(  # IR_PLANET
            x=[float(epochs_yr[idx])],
            y=[rv[idx]],
            mode="markers",
            marker=dict(size=10, color="violet", line=dict(color="black", width=1)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#333333", size=13, family="system-ui, sans-serif"),
        margin=dict(t=30, b=55, l=70, r=20),
        xaxis=dict(
            title="time [yr]",
            range=[float(epochs_yr[0]), float(epochs_yr[-1])],
            gridcolor="#e8e8e8",
            zerolinecolor="#bbbbbb",
        ),
        yaxis=dict(
            title="Stellar RV [km s⁻¹]",
            range=[-1.5 * maxK, 1.5 * maxK],
            gridcolor="#e8e8e8",
            zerolinecolor="#bbbbbb",
        ),
    )
    return fig


# ── App layout ───────────────────────────────────────────────────────────────

BG = "#ffffff"  # page background (left side)
PANEL_BG = "#1a1a2e"  # dark sidebar — matches the plot backgrounds
LIGHT = "#da70d6"  # orchid, for headings and labels
DIM = "#8899aa"  # muted text

_initial_orbit = compute_orbit(0, 0, 0, 0)
_initial_orbit_fig = build_orbit_figure(
    _initial_orbit[0], _initial_orbit[1], _initial_orbit[3], pan_deg=0, idx=0
)
_initial_rv_fig = build_rv_figure(_initial_orbit[2], idx=0)

app = dash.Dash(__name__, title="Show Me the Orbit", update_title=None)

_MARK_STYLE = {"color": DIM, "fontSize": "10px"}


def slider_row(label, id_, min_, max_, step, value, marks):
    styled_marks = {
        k: {"label": str(v), "style": _MARK_STYLE} for k, v in marks.items()
    }
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "color": LIGHT,
                    "fontSize": "10px",
                    "fontFamily": "system-ui, sans-serif",
                    "fontWeight": "700",
                    "letterSpacing": "0.05em",
                    "display": "block",
                    "marginBottom": "10px",
                },
            ),
            dcc.Slider(
                id=id_,
                min=min_,
                max=max_,
                step=step,
                value=value,
                marks=styled_marks,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
        ],
        style={"marginBottom": "28px"},
    )


def info_section(heading, body):
    """A heading + paragraph block for the explanatory sidebar text."""
    return html.Div(
        [
            html.P(
                heading,
                style={
                    "color": LIGHT,
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "letterSpacing": "0.09em",
                    "textTransform": "uppercase",
                    "marginBottom": "6px",
                    "marginTop": "20px",
                },
            ),
            html.Div(
                body,
                style={
                    "color": DIM,
                    "fontSize": "14px",
                    "lineHeight": "1.7",
                    "marginTop": 0,
                },
            ),
        ]
    )


# Vertical budget for the left column (all in px, matching the padding/gap below):
#   top padding 20 + bottom padding 20 + gap 16 + rv height 180 = 236
# So orbit height = calc(100vh - 236px), which fills the rest exactly.
_RV_H = "180px"
_ORBIT_H = "calc(100vh - 236px)"  # 236 = 20+20 padding + 16 gap + 180 rv

app.layout = html.Div(
    style={
        "height": "100vh",
        "fontFamily": "system-ui, sans-serif",
        "overflow": "hidden",
    },
    children=[
        # ── Two-column grid ───────────────────────────────────────────────────
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 300px",
                "height": "100vh",
            },
            children=[
                # ── Left: plots ───────────────────────────────────────────────
                html.Div(
                    style={
                        "backgroundColor": BG,
                        "padding": "20px",
                        "boxSizing": "border-box",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px",
                        "height": "100vh",
                        "overflow": "hidden",
                    },
                    children=[
                        dcc.Graph(
                            id="orbit-plot",
                            figure=_initial_orbit_fig,
                            responsive=True,
                            config={"displayModeBar": False},
                            # height drives the square; maxWidth prevents overflow on
                            # narrow viewports (scaleanchor keeps data aspect ratio)
                            style={
                                "height": _ORBIT_H,
                                "aspectRatio": "1 / 1",
                                "maxWidth": "100%",
                                "flexShrink": "0",
                            },
                        ),
                        dcc.Graph(
                            id="rv-plot",
                            figure=_initial_rv_fig,
                            responsive=True,
                            config={"displayModeBar": False},
                            style={"height": _RV_H, "width": "100%", "flexShrink": "0"},
                        ),
                    ],
                ),
                # ── Right: dark sidebar ───────────────────────────────────────
                html.Div(
                    style={
                        "backgroundColor": PANEL_BG,
                        "padding": "24px",
                        "boxSizing": "border-box",
                        "height": "100vh",
                        "overflowY": "auto",
                    },
                    children=[
                        html.H1(
                            "Show Me the Orbit",
                            style={
                                "color": "#ffffff",
                                "fontSize": "22px",
                                "fontWeight": "normal",
                                "marginTop": 0,
                                "marginBottom": "4px",
                                "lineHeight": "1.3",
                            },
                        ),
                        html.P(
                            "Sarah Blunt (2019; web port 2026)",
                            style={
                                "color": DIM,
                                "fontSize": "11px",
                                "marginTop": 0,
                                "marginBottom": "28px",
                            },
                        ),
                        html.Hr(
                            style={"borderColor": "#2a3a5a", "marginBottom": "24px"}
                        ),
                        slider_row(
                            "e — eccentricity",
                            "ecc-slider",
                            0,
                            MAX_ECC,
                            0.05,
                            0,
                            {0: "0", 0.5: "0.5", 0.95: "0.95"},
                        ),
                        slider_row(
                            "i — inclination (°)",
                            "inc-slider",
                            0,
                            180,
                            10,
                            0,
                            {0: "0", 90: "90", 180: "180"},
                        ),
                        slider_row(
                            "ω — arg. of periastron (°)",
                            "aop-slider",
                            0,
                            360,
                            10,
                            0,
                            {0: "0", 180: "180", 360: "360"},
                        ),
                        slider_row(
                            "Ω — pos. angle of nodes (°)",
                            "pan-slider",
                            0,
                            360,
                            10,
                            0,
                            {0: "0", 180: "180", 360: "360"},
                        ),
                        html.Hr(
                            style={"borderColor": "#2a3a5a", "marginBottom": "4px"}
                        ),
                        info_section(
                            "The Orbital Elements in Words",
                            html.Ul(
                                [
                                    html.Li(
                                        [
                                            html.B("ω"),
                                            " is the argument of periastron. "
                                            "This parameter rotates the vector connecting "
                                            "the star and the planet's periastron ",
                                            html.B("within the orbital plane"),
                                            ".",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("Ω"),
                                            " is the position angle of nodes. "
                                            "This parameter rotates the line of nodes "
                                            "within the plane of the sky.",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("i"),
                                            " is the orbit's inclination. "
                                            "This parameter rotates the orbital plane about "
                                            "its line of nodes (dashed black line in top panel).",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("e"),
                                            " is the orbit's eccentricity. "
                                            "This parameter controls the orbit's shape and "
                                            "the planet's speed over time.",
                                        ]
                                    ),
                                ],
                                style={
                                    "paddingLeft": "18px",
                                    "marginTop": "4px",
                                    "marginBottom": 0,
                                },
                            ),
                        ),
                        info_section(
                            "Symbol and Color Definitions",
                            html.Ul(
                                [
                                    html.Li(
                                        [
                                            "The pink circle tracks the planet's position, "
                                            "and the pink star shows the primary's location. "
                                            "The top panel is shown in the reference frame "
                                            "where the star is stationary. In other words, "
                                            "the orbital track plotted shows the ",
                                            html.B("relative RA and Dec"),
                                            " of the planet and star over time.",
                                        ]
                                    ),
                                    html.Li(
                                        "The darker portion of the orbital arc is in front "
                                        "of the plane of the sky (closer to Earth), and the "
                                        "lighter portion is behind the plane of the sky "
                                        "(farther from Earth). When the planet symbol becomes "
                                        "transparent, the planet is behind the plane of the sky."
                                    ),
                                    html.Li(
                                        "The black dashed line in the top panel is the line "
                                        "of nodes, or the line in the plane of the sky that "
                                        "intersects the orbit."
                                    ),
                                ],
                                style={
                                    "paddingLeft": "18px",
                                    "marginTop": "4px",
                                    "marginBottom": 0,
                                },
                            ),
                        ),
                        info_section(
                            "A Caution to RV Astronomers",
                            html.Ul(
                                [
                                    html.Li(
                                        [
                                            "orbitize! defines ω to be the argument of "
                                            "periastron of the ",
                                            html.B("planet's"),
                                            " orbit, whereas most RV codes (e.g. radvel) "
                                            "define ω to be the argument of periastron of "
                                            "the ",
                                            html.B("star's"),
                                            " orbit. In practice, this means that you need "
                                            "to add 180° to ω when comparing the outputs of "
                                            "RV codes and orbitize!.",
                                        ]
                                    ),
                                ],
                                style={
                                    "paddingLeft": "18px",
                                    "marginTop": "4px",
                                    "marginBottom": 0,
                                },
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # ── Hidden state (outside the grid so it doesn't create extra rows) ──
        dcc.Interval(id="interval", interval=50, n_intervals=0),
        dcc.Store(
            id="orbit-cache",
            data={
                "ra": _initial_orbit[0],
                "dec": _initial_orbit[1],
                "rv": _initial_orbit[2],
                "z": _initial_orbit[3],
            },
        ),
        dcc.Store(id="anim-state", data={"idx": 0}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────


app.clientside_callback(
    f"""
    function(ecc, inc_deg, aop_deg, pan_deg, state) {{
        var SMA      = {SMA};
        var TAU      = {TAU};
        var PLX      = {PLX};
        var MTOT     = {MTOT};
        var MPLANET  = {MPLANET};
        var N        = {TIME_DIVS};
        var MAX_SEP  = {float(max_sep):.6f};

        var per_yr    = Math.sqrt(SMA * SMA * SMA / MTOT);
        var inc = inc_deg * Math.PI / 180;
        var aop = aop_deg * Math.PI / 180;
        var pan = pan_deg * Math.PI / 180;

        var ca = Math.cos(aop), sa = Math.sin(aop);
        var cp = Math.cos(pan), sp = Math.sin(pan);
        var ci = Math.cos(inc), si = Math.sin(inc);

        // Thiele-Innes constants (sky-plane projection)
        var TI_A =  ca*cp - sa*sp*ci;
        var TI_B =  ca*sp + sa*cp*ci;
        var TI_F = -sa*cp - ca*sp*ci;
        var TI_G = -sa*sp + ca*cp*ci;

        // RV semi-amplitude [km/s]; 1 AU/yr = 4.74047 km/s
        var sqrt1e2 = Math.sqrt(1 - ecc*ecc);
        var K = 2*Math.PI * 4.74047 * MPLANET * si / (Math.sqrt(MTOT * SMA) * sqrt1e2);

        var ra = new Array(N), dec = new Array(N);
        var rv = new Array(N), z  = new Array(N);

        for (var i = 0; i < N; i++) {{
            var t_yr = (i / (N - 1)) * per_yr;

            // Mean anomaly, wrapped to [0, 2π]
            var phase = (t_yr / per_yr - TAU) % 1.0;
            if (phase < 0) phase += 1.0;
            var M = 2 * Math.PI * phase;

            // Eccentric anomaly via Newton-Raphson
            var E = M;
            for (var it = 0; it < 50; it++) {{
                var dE = (M - E + ecc * Math.sin(E)) / (1 - ecc * Math.cos(E));
                E += dE;
                if (Math.abs(dE) < 1e-10) break;
            }}

            var cosE = Math.cos(E), sinE = Math.sin(E);

            // Position in orbital plane
            var X = SMA * (cosE - ecc);
            var Y = SMA * sqrt1e2 * sinE;

            // Sky coordinates [mas]
            ra[i]  = PLX * (TI_B * X + TI_G * Y);
            dec[i] = PLX * (TI_A * X + TI_F * Y);

            // RV [km/s]: stellar motion (sign flip from planet aop convention)
            var nu = Math.atan2(sqrt1e2 * sinE, cosE - ecc);
            rv[i] = -K * (Math.cos(nu + aop) + ecc * Math.cos(aop));

            // z-depth via base orbit (aop0=pan0=0 ⟹ ra0=PLX·Y, dec0=PLX·X)
            z[i] = si * (-ca * PLX * Y - sa * PLX * X) * PLX;
        }}

        // Update figures via Plotly.restyle (no server round-trip)
        var idx = (state && state.idx != null) ? state.idx : 0;
        var epoch_yr = (idx / (N - 1)) * per_yr;

        var front_ra  = ra.map( function(r,i) {{ return z[i] >= 0 ? r : null; }});
        var front_dec = dec.map(function(d,i) {{ return z[i] >= 0 ? d : null; }});
        var back_ra   = ra.map( function(r,i) {{ return z[i] <  0 ? r : null; }});
        var back_dec  = dec.map(function(d,i) {{ return z[i] <  0 ? d : null; }});
        var lon_x = [-Math.sin(pan)*MAX_SEP, Math.sin(pan)*MAX_SEP];
        var lon_y = [-Math.cos(pan)*MAX_SEP, Math.cos(pan)*MAX_SEP];
        var op = z[idx] >= 0 ? 1.0 : 0.15;

        function gd(id) {{
            var el = document.getElementById(id);
            if (!el) return null;
            return el._fullLayout ? el : el.querySelector('.js-plotly-plot');
        }}

        var od = gd('orbit-plot');
        if (od) {{
            Plotly.restyle(od, {{x: [back_ra],  y: [back_dec]}},  [{IO_BACK}]);
            Plotly.restyle(od, {{x: [front_ra], y: [front_dec]}}, [{IO_FRONT}]);
            Plotly.restyle(od, {{x: [lon_x],    y: [lon_y]}},     [{IO_NODES}]);
            Plotly.restyle(od, {{x: [[ra[idx]]], y: [[dec[idx]]], 'marker.opacity': [op]}}, [{IO_PLANET}]);
        }}

        var rd = gd('rv-plot');
        if (rd) {{
            Plotly.restyle(rd, {{y: [rv]}},                              [{IR_TRACK}]);
            Plotly.restyle(rd, {{x: [[epoch_yr]], y: [[rv[idx]]]}},      [{IR_PLANET}]);
        }}

        return [{{ra: ra, dec: dec, rv: rv, z: z}}, {{idx: idx}}];
    }}
    """,
    Output("orbit-cache", "data"),
    Output("anim-state", "data"),
    Input("ecc-slider", "value"),
    Input("inc-slider", "value"),
    Input("aop-slider", "value"),
    Input("pan-slider", "value"),
    State("anim-state", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    f"""
    function(n_intervals, state, cache) {{
        if (!cache || !state) return window.dash_clientside.no_update;
        var idx = (state.idx + 1) % {TIME_DIVS};
        var epoch_yr = idx / {TIME_DIVS - 1} * {float(epochs_yr[-1]):.6f};

        function plotlyDiv(id) {{
            var el = document.getElementById(id);
            if (!el) return null;
            return el._fullLayout ? el : el.querySelector('.js-plotly-plot');
        }}

        var orbitDiv = plotlyDiv('orbit-plot');
        if (orbitDiv) {{
            Plotly.restyle(orbitDiv, {{
                x: [[cache.ra[idx]]],
                y: [[cache.dec[idx]]],
                'marker.opacity': [cache.z[idx] >= 0 ? 1.0 : 0.15]
            }}, [{IO_PLANET}]);
        }}

        var rvDiv = plotlyDiv('rv-plot');
        if (rvDiv) {{
            Plotly.restyle(rvDiv, {{
                x: [[epoch_yr]],
                y: [[cache.rv[idx]]]
            }}, [{IR_PLANET}]);
        }}

        return {{idx: idx}};
    }}
    """,
    Output("anim-state", "data", allow_duplicate=True),
    Input("interval", "n_intervals"),
    State("anim-state", "data"),
    State("orbit-cache", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=True)
