import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.common import paths
from src.common.paths import (COUNTRIES_GEOJSON, METRICS, CROP_ANOMALIES, CLIMATE, EVENTS, SOURCES_MD, EXPLORER,
                              USE_DUMMY, WET_EVENTS, METRICS_WET, TRADE)
from src.metrics.portfolio import evaluate

MAX_SIZE_MB = 15
MAX_SLIDERS = 12
PORTFOLIO_COV = paths.OUT / "portfolio_cov.json"

# Optional phase-2 map layers: key -> (label, description, colorscale)
EXTRA_METRICS = {
    'vuln_wet': ('Wet-year Vulnerability (% years < 0.9)',
                 'The share of wet years (SPEI-12 of +1.0 or higher) in which national agricultural value is more than 10% below its long-term trend.',
                 'Blues'),
    'resistance_wet': ('Wet-event Resistance (Median ratio)',
                       'Median ratio of national agricultural value in the first year of a wet event to the mean of the three preceding years (same formula as Resistance, applied to wet events).',
                       'YlGnBu'),
    'import_dep': ('Cereal Import Dependency',
                   'Mean of cereal imports divided by cereal domestic supply (production + imports - exports), 1993-2023 (FAOSTAT Food Balances).',
                   'Oranges'),
}


def _r(x, sig=6):
    """Round to `sig` significant digits; NaN/None -> None."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{sig}g}") if np.isfinite(x) else None


def _clean(v):
    """NaN -> None for JSON."""
    if v is None:
        return None
    try:
        return None if not np.isfinite(float(v)) else float(v)
    except (TypeError, ValueError):
        return v


def load_portfolios(valid_iso3):
    """Round portfolio_cov.json for embedding; assert rounded arrays still reproduce evaluate() at w0."""
    if not PORTFOLIO_COV.exists():
        print(f"  Portfolio file not found ({PORTFOLIO_COV}); skipping the portfolio simulator data.")
        return {}
    raw = json.loads(PORTFOLIO_COV.read_text(encoding="utf-8"))
    meta, countries = raw["meta"], raw["countries"]
    out = {}
    for iso3, c in countries.items():
        if iso3 not in valid_iso3:
            continue
        p = {
            'items': c['items'],
            'w0': [_r(x, 8) for x in c['w0']],
            'lo': [_r(x, 8) for x in c['lo']],
            'hi': [_r(x, 8) for x in c['hi']],
            'mu': [_r(x) for x in c['mu']],
            'cov': [[_r(x) for x in row] for row in c['cov']],
            'beta': [_r(x) for x in c['beta']],
            'frontier': [{'exp_value_ha': _r(f['exp_value_ha']), 'risk_sd': _r(f['risk_sd'])} for f in c['frontier']],
        }
        n = len(p['items'])
        assert len(p['w0']) == len(p['mu']) == len(p['cov']) == len(p['lo']) == len(p['hi']) == len(p['beta']) == n, iso3
        assert all(len(row) == n for row in p['cov']), iso3
        assert abs(sum(p['w0']) - 1) < 1e-6, (iso3, sum(p['w0']))
        # rounded embedded arrays must reproduce evaluate() on the original arrays
        nan = lambda a: [np.nan if x is None else x for x in a]
        ref = evaluate(nan(c['w0']), nan(c['mu']), c['cov'], nan(c['beta']))
        got = evaluate(nan(p['w0']), nan(p['mu']), p['cov'], nan(p['beta']))
        for k in ('exp_value_ha', 'risk_sd', 'drought_change'):
            if np.isnan(ref[k]):
                assert np.isnan(got[k]), (iso3, k)
            else:
                assert abs(got[k] - ref[k]) <= 1e-4 * max(abs(ref[k]), 1e-12), (iso3, k, got[k], ref[k])
        out[iso3] = p
    return out

# Plain (non f-string) blocks spliced into the template, so JS braces are written singly.
PF_CSS = """
        #tabs { display: flex; gap: 6px; padding: 8px 10px 0 10px; background: #fafafa; border-bottom: 1px solid #ccc; }
        .tab-btn { padding: 8px 16px; font-size: 15px; cursor: pointer; border: 1px solid #ccc; border-bottom: none; background: #ecf0f1; border-radius: 4px 4px 0 0; }
        .tab-btn.active { background: #fff; font-weight: bold; border-color: #2c3e50; }
        #pf-view { display: none; flex: 1; padding: 10px; overflow-y: auto; background: #fafafa; font-size: 14px; }
        #pf-table { border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 10px; }
        #pf-table th, #pf-table td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; font-size: 14px; }
        #pf-table th:first-child, #pf-table td:first-child { text-align: left; }
        #pf-chart { height: 320px; background: #fff; border: 1px solid #eee; border-radius: 4px; margin-bottom: 10px; }
        #pf-sliders { background: #fff; border: 1px solid #eee; border-radius: 4px; padding: 8px 12px; margin-bottom: 10px; }
        .pf-row { display: grid; grid-template-columns: 170px 1fr 70px; align-items: center; gap: 8px; margin: 4px 0; font-size: 14px; }
        .pf-row input[type=range] { width: 100%; }
        .pf-row span.pf-val { text-align: right; font-variant-numeric: tabular-nums; }
        #pf-reset { padding: 8px 14px; font-size: 14px; cursor: pointer; margin-bottom: 8px; }
        #pf-note { font-size: 14px; color: #444; background: #fff; border: 1px solid #ddd; border-radius: 4px; padding: 10px; line-height: 1.5; }
        #pf-fixed { font-size: 14px; color: #555; margin-bottom: 10px; }
"""

PF_HTML = """
        <div id="pf-view">
            <div id="pf-empty" style="font-size:16px;color:#555;">Click a country on the map to open its portfolio simulator.</div>
            <div id="pf-body" style="display:none;">
                <strong id="pf-title"></strong>
                <table id="pf-table">
                    <thead><tr><th>Measure</th><th>Current mix</th><th>Your mix</th><th>Difference</th></tr></thead>
                    <tbody id="pf-tbody"></tbody>
                </table>
                <div id="pf-chart"></div>
                <button id="pf-reset">Reset to current</button>
                <div id="pf-sliders"></div>
                <div id="pf-fixed"></div>
                <div id="pf-note">
                    <strong>Model assumptions.</strong> Prices are constant (2014-2016 international dollars). A crop's value per hectare
                    and its year-to-year variability are assumed unchanged when its area changes. Each crop's area share is bounded to
                    0.5 to 1.5 times its current share and to within 10 percentage points of it. Only crops the country already grows are
                    included, and area outside these crops is not modelled. Risk is the standard deviation across years of the relative
                    shortfall of total value per hectare. The drought figure is the relative change in total value in a year with SPEI-12 = -1.5
                    compared with a year with SPEI-12 = 0, from each crop's drought response slope; it is shown as n/a if any crop with a
                    non-zero share has no slope.
                </div>
            </div>
        </div>
"""

PF_JS = """
        // ---------------- Portfolio simulator ----------------
        const SPEI_DROUGHT = -1.5;
        const MAX_SLIDERS = __MAX_SLIDERS__;

        // BEGIN evaluate (same formulas as src.metrics.portfolio.evaluate)
        function evaluatePortfolio(w, mu, cov, beta) {
            const n = w.length;
            let e = 0;
            for (let i = 0; i < n; i++) e += w[i] * mu[i];
            const s = new Array(n);
            for (let i = 0; i < n; i++) s[i] = w[i] * mu[i] / e;
            let q = 0;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) q += s[i] * cov[i][j] * s[j];
            }
            const risk = Math.sqrt(Math.max(q, 0));
            let complete = true, sb = 0;
            for (let i = 0; i < n; i++) {
                if (beta[i] === null) { if (w[i] > 0) complete = false; }
                else sb += s[i] * beta[i];
            }
            return { exp_value_ha: e, risk_sd: risk, drought_change: complete ? SPEI_DROUGHT * sb : null };
        }
        // END evaluate

        // BEGIN reallocate
        // Move adjustable crop k to x (fraction); absorb the change in the other adjustable crops
        // proportionally to their current shares, respecting [lo, hi]. adj = indices of adjustable crops,
        // T = total share of the adjustable crops. Returns a new weight array.
        function reallocate(w, k, x, adj, lo, hi, T) {
            const others = adj.filter(i => i !== k);
            let loSum = 0, hiSum = 0;
            others.forEach(i => { loSum += lo[i]; hiSum += hi[i]; });
            const xmin = Math.max(lo[k], T - hiSum), xmax = Math.min(hi[k], T - loSum);
            x = Math.min(Math.max(x, xmin), xmax);
            const nw = w.slice();
            nw[k] = x;
            let free = others.slice();
            for (let iter = 0; iter < 100 && free.length > 0; iter++) {
                let cur = 0;
                others.forEach(i => { cur += nw[i]; });
                const rem = T - x - cur;
                if (Math.abs(rem) < 1e-12) break;
                let tot = 0;
                free.forEach(i => { tot += nw[i]; });
                free.forEach(i => {
                    const wt = tot > 1e-12 ? nw[i] / tot : 1 / free.length;
                    nw[i] = Math.min(Math.max(nw[i] + rem * wt, lo[i]), hi[i]);
                });
                free = free.filter(i => nw[i] > lo[i] + 1e-12 && nw[i] < hi[i] - 1e-12);
            }
            return nw;
        }
        // END reallocate

        let pf = null;   // state of the displayed portfolio
        let activeTab = 'ts';

        function signed(v, d, unit) { return (v >= 0 ? '+' : '') + v.toFixed(d) + unit; }

        function pfRows(cur, usr) {
            const pctDiff = (a, b) => (b - a) / a * 100;
            const rows = [];
            rows.push(['Expected value per ha (I$/ha)', cur.exp_value_ha.toFixed(0), usr.exp_value_ha.toFixed(0),
                       signed(pctDiff(cur.exp_value_ha, usr.exp_value_ha), 2, '%')]);
            rows.push(['Risk (SD of relative shortfall, %)', (cur.risk_sd * 100).toFixed(2), (usr.risk_sd * 100).toFixed(2),
                       signed(pctDiff(cur.risk_sd, usr.risk_sd), 2, '%')]);
            if (cur.drought_change === null || usr.drought_change === null) {
                rows.push(['Value change at SPEI-12 = -1.5 (%)', 'n/a', 'n/a', 'n/a']);
            } else {
                rows.push(['Value change at SPEI-12 = -1.5 (%)', (cur.drought_change * 100).toFixed(2), (usr.drought_change * 100).toFixed(2),
                           signed((usr.drought_change - cur.drought_change) * 100, 2, ' pp')]);
            }
            return rows;
        }

        function pfUpdateOutputs(redrawChart) {
            const p = pf.p;
            const cur = evaluatePortfolio(p.w0, p.mu, p.cov, p.beta);
            const usr = evaluatePortfolio(pf.w, p.mu, p.cov, p.beta);
            document.getElementById('pf-tbody').innerHTML = pfRows(cur, usr)
                .map(r => '<tr><td>' + r.join('</td><td>') + '</td></tr>').join('');
            if (redrawChart) {
                const fx = p.frontier.map(f => f.risk_sd * 100), fy = p.frontier.map(f => f.exp_value_ha);
                Plotly.newPlot('pf-chart', [
                    { x: fx, y: fy, mode: 'lines+markers', name: 'Frontier within bounds', line: { color: '#7f8c8d' }, marker: { size: 5 } },
                    { x: [cur.risk_sd * 100], y: [cur.exp_value_ha], mode: 'markers', name: 'Current mix', marker: { size: 14, color: '#2c3e50', symbol: 'diamond' } },
                    { x: [usr.risk_sd * 100], y: [usr.exp_value_ha], mode: 'markers', name: 'Your mix', marker: { size: 14, color: '#e67e22' } }
                ], {
                    margin: { t: 30, b: 50, l: 70, r: 20 }, font: { size: 14 },
                    title: { text: 'Frontier: expected value vs risk', font: { size: 15 } },
                    xaxis: { title: 'Risk: SD of relative shortfall (%)' },
                    yaxis: { title: 'Expected value (I$/ha)' },
                    legend: { orientation: 'h', y: -0.25 }
                }, { responsive: true });
            } else {
                Plotly.restyle('pf-chart', { x: [[usr.risk_sd * 100]], y: [[usr.exp_value_ha]] }, [2]);
            }
        }

        function pfSyncSliders() {
            pf.adj.forEach(i => {
                document.getElementById('pf-slider-' + i).value = (pf.w[i] * 100).toFixed(2);
                document.getElementById('pf-val-' + i).innerText = (pf.w[i] * 100).toFixed(1) + '%';
            });
        }

        function renderPortfolio() {
            if (!currentCountry) return;
            const empty = document.getElementById('pf-empty');
            const body = document.getElementById('pf-body');
            empty.style.display = 'none';
            const p = portfolioData[currentCountry];
            if (!p) {
                body.style.display = 'none';
                empty.style.display = 'block';
                empty.innerText = 'No portfolio for this country (' + currentCountry + ').';
                pf = null;
                return;
            }
            body.style.display = 'block';
            const order = p.w0.map((v, i) => i).sort((a, b) => p.w0[b] - p.w0[a]);
            const adj = order.slice(0, MAX_SLIDERS);
            const fixed = order.slice(MAX_SLIDERS);
            let T = 0;
            adj.forEach(i => { T += p.w0[i]; });
            pf = { p: p, w: p.w0.slice(), adj: adj, T: T };
            document.getElementById('pf-title').innerText = currentCountry + ' - area shares of ' + p.items.length + ' crops';
            document.getElementById('pf-sliders').innerHTML = adj.map(i =>
                '<div class="pf-row"><label for="pf-slider-' + i + '">' + p.items[i] + '</label>' +
                '<input type="range" id="pf-slider-' + i + '" min="' + (p.lo[i] * 100) + '" max="' + (p.hi[i] * 100) +
                '" step="0.05" value="' + (p.w0[i] * 100) + '"><span class="pf-val" id="pf-val-' + i + '"></span></div>').join('');
            adj.forEach(i => {
                document.getElementById('pf-slider-' + i).addEventListener('input', function(e) {
                    pf.w = reallocate(pf.w, i, parseFloat(e.target.value) / 100, pf.adj, pf.p.lo, pf.p.hi, pf.T);
                    pfSyncSliders();
                    pfUpdateOutputs(false);
                });
            });
            document.getElementById('pf-fixed').innerText = fixed.length === 0 ? '' :
                'Sliders are shown for the ' + MAX_SLIDERS + ' largest crops. The other ' + fixed.length +
                ' crops are held at their current shares: ' + fixed.map(i => p.items[i] + ' (' + (p.w0[i] * 100).toFixed(1) + '%)').join(', ') + '.';
            pfSyncSliders();
            pfUpdateOutputs(true);
        }

        document.getElementById('pf-reset').addEventListener('click', function() {
            if (!pf) return;
            pf.w = pf.p.w0.slice();
            pfSyncSliders();
            pfUpdateOutputs(false);
        });

        function showTab(name) {
            activeTab = name;
            document.getElementById('tab-ts').classList.toggle('active', name === 'ts');
            document.getElementById('tab-pf').classList.toggle('active', name === 'pf');
            document.getElementById('charts-inner').style.display = name === 'ts' ? 'flex' : 'none';
            document.getElementById('pf-view').style.display = name === 'pf' ? 'block' : 'none';
            if (name === 'pf') {
                renderPortfolio();
            } else if (currentCountry) {
                Plotly.Plots.resize('line-chart');
                Plotly.Plots.resize('bar-chart');
            }
        }
        document.getElementById('tab-ts').addEventListener('click', () => showTab('ts'));
        document.getElementById('tab-pf').addEventListener('click', () => showTab('pf'));
"""


def build_explorer():
    print("Building interactive explorer...")
    
    # 1. Load data
    print("  Loading data...")
    metrics_df = pd.read_parquet(METRICS)
    anom_df = pd.read_parquet(CROP_ANOMALIES)
    climate_df = pd.read_parquet(CLIMATE)
    events_df = pd.read_csv(EVENTS)
    gdf = gpd.read_file(COUNTRIES_GEOJSON)
    
    try:
        with open(SOURCES_MD, "r") as f:
            sources_md_content = f.read()
    except FileNotFoundError:
        sources_md_content = "SOURCES.md not found in data/raw/"

    # Optional phase-2 inputs
    extra_values = {}   # key -> {iso3: value}
    if METRICS_WET.exists():
        mw = pd.read_parquet(METRICS_WET).set_index('iso3')
        for key in ('vuln_wet', 'resistance_wet'):
            extra_values[key] = {i: _clean(v) for i, v in mw[key].items()}
    else:
        print(f"  {METRICS_WET} not found: skipping vuln_wet and resistance_wet.")
    if TRADE.exists():
        tr = pd.read_parquet(TRADE)
        extra_values['import_dep'] = {i: _clean(v) for i, v in tr.groupby('iso3')['import_dep'].mean().items()}
    else:
        print(f"  {TRADE} not found: skipping import_dep.")
    if WET_EVENTS.exists():
        wet_df = pd.read_csv(WET_EVENTS)
    else:
        print(f"  {WET_EVENTS} not found: no wet-event shading.")
        wet_df = pd.DataFrame(columns=['iso3', 'year'])
    extra_metrics = {k: v for k, v in EXTRA_METRICS.items() if k in extra_values}

    # 2. Filter to included countries
    included_iso3 = metrics_df['iso3'].unique()
    
    # 3. Prepare GeoJSON
    # Merge metrics into geojson properties
    print("  Preparing GeoJSON...")
    gdf = gdf[gdf['iso3'].isin(included_iso3)].copy()
    
    # We need to map columns from metrics_df to gdf
    metrics_dict = metrics_df.set_index('iso3').to_dict('index')
    
    # We will build a new lightweight geojson dict to keep size small
    geojson_data = json.loads(gdf.to_json())
    for feature in geojson_data['features']:
        iso3 = feature['properties'].get('iso3')
        if iso3 in metrics_dict:
            # Update properties with metrics
            feature['properties'].update(metrics_dict[iso3])
            for key, vals in extra_values.items():
                feature['properties'][key] = vals.get(iso3)
            
            # Format nicely for hover
            m = metrics_dict[iso3]
            feature['properties']['hover_text'] = (
                f"<b>{iso3}</b><br>"
                f"Hill1: {m.get('hill1', 0):.2f}<br>"
                f"Resp Div: {m.get('resp_div', 0):.3f}<br>"
                f"Stability: {m.get('stability', 0):.2f}<br>"
                f"Events: {m.get('n_events', 0)}"
            )

    # 4. Prepare timeseries data for the right panel
    print("  Preparing timeseries data...")
    ts_data = {}
    for iso3 in included_iso3:
        # Climate
        c_df = climate_df[climate_df['iso3'] == iso3].sort_values('year')
        years = c_df['year'].tolist()
        spei = c_df['spei12_w'].tolist()
        
        # Events
        e_df = events_df[events_df['iso3'] == iso3]
        drought_years = e_df['year'].tolist()
        wet_years = wet_df[wet_df['iso3'] == iso3]['year'].tolist()
        
        # Top 6 crops by mean value_share
        a_df = anom_df[anom_df['iso3'] == iso3]
        if a_df.empty:
            continue
            
        crop_shares = a_df.groupby('item')['value_share'].mean().sort_values(ascending=False)
        top_crops = crop_shares.head(6).index.tolist()
        
        crops_data = {}
        for crop in top_crops:
            crop_df = a_df[a_df['item'] == crop].set_index('year')
            # align with global years
            aligned = crop_df.reindex(years)['yield_anom'].fillna(0).tolist()
            crops_data[crop] = aligned
            
        ts_data[iso3] = {
            'years': years,
            'spei': spei,
            'drought_years': drought_years,
            'wet_years': wet_years,
            'crops': crops_data
        }

    portfolios = load_portfolios(set(ts_data))
    print(f"  Portfolios embedded: {len(portfolios)}")
    extra_options = "".join(f'<option value="{k}">{v[0]}</option>' for k, v in extra_metrics.items())

    # 5. HTML Template
    print("  Generating HTML...")
    
    PF_JS_FILLED = PF_JS.replace("__MAX_SLIDERS__", str(MAX_SLIDERS))
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Crop Diversity & Drought Resilience Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }}
        #header {{ background: #2c3e50; color: white; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }}
        #header h1 {{ margin: 0; font-size: 20px; }}
        #header button {{ background: #34495e; color: white; border: 1px solid #7f8c8d; padding: 8px 15px; cursor: pointer; border-radius: 4px; }}
        #header button:hover {{ background: #7f8c8d; }}
        #main {{ display: flex; flex: 1; overflow: hidden; }}
        #map-container {{ flex: 1; border-right: 1px solid #ccc; display: flex; flex-direction: column; }}
        #controls-wrapper {{ padding: 15px; background: #ecf0f1; border-bottom: 1px solid #bdc3c7; display: flex; flex-direction: column; gap: 10px; }}
        #controls {{ display: flex; align-items: center; gap: 10px; }}
        #controls select {{ padding: 8px; font-size: 14px; border-radius: 4px; border: 1px solid #bdc3c7; }}
        #metric-description {{ font-size: 13px; color: #555; background: #fff; padding: 10px; border-radius: 4px; border: 1px solid #ddd; }}
        #map {{ flex: 1; }}
        #charts-container {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #fafafa; }}
        .chart-wrapper {{ flex: 1; min-height: 300px; margin-bottom: 10px; background: white; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }}
        .modal-content {{ background: white; margin: 5% auto; padding: 30px; width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); line-height: 1.6; }}
        .close {{ float: right; cursor: pointer; font-size: 28px; font-weight: bold; color: #7f8c8d; margin-top: -10px; }}
        .close:hover {{ color: #2c3e50; }}
        
        .instruction-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255,255,255,0.9); padding: 20px; border-radius: 8px; border: 1px solid #ddd; text-align: center; pointer-events: none; z-index: 10; font-size: 18px; color: #555; }}
        #charts-inner {{ position: relative; width: 100%; flex: 1; box-sizing: border-box; padding: 10px; overflow-y: auto; display: flex; flex-direction: column; }}
        {PF_CSS}
        #chart-explanation {{ display: none; padding: 12px; background: #e8f4f8; border-left: 4px solid #3498db; margin-bottom: 10px; font-size: 14px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Crop Diversity & Drought Resilience Explorer</h1>
        <button onclick="document.getElementById('about-modal').style.display='block'">About the Data</button>
    </div>
    
    <div id="main">
        <div id="map-container">
            <div id="controls-wrapper">
                <div id="controls">
                    <label for="metric-select"><strong>Color Map by:</strong></label>
                    <select id="metric-select">
                        <option value="hill1">Crop Diversity (hill1)</option>
                        <option value="resp_div">Response Diversity</option>
                        <option value="stability">Stability</option>
                        <option value="resistance">Resistance</option>
                        <option value="recovery">Recovery</option>
                        <option value="vulnerability">Vulnerability</option>
                        <option value="phi_sync">Synchrony (phi_sync)</option>
                        {extra_options}
                    </select>
                </div>
                <div id="metric-description"></div>
            </div>
            <div id="map"></div>
        </div>
        
        <div id="charts-container">
            <div id="tabs">
                <button id="tab-ts" class="tab-btn active">Time series</button>
                <button id="tab-pf" class="tab-btn">Portfolio simulator</button>
            </div>
            <div id="charts-inner">
                <div id="instruction" class="instruction-overlay">Click a country on the map to view timeseries</div>
                <div id="chart-explanation">
                    <strong>Top Chart:</strong> Shows the yield anomalies (deviations from long-term trends) for the country's top 6 crops by value.<br>
                    <strong>Bottom Chart:</strong> Shows the Cropland-weighted SPEI-12 climate index. Negative values indicate drier conditions.<br>
                    <strong>Red Shaded Regions:</strong> Represent verified drought events (SPEI-12 &le; -1.0) based on the project's criteria.<br>
                    <strong>Blue Shaded Regions:</strong> Represent wet events (SPEI-12 &ge; +1.0), the mirror of the drought rule. Bars are red when SPEI-12 &le; -1.0 and dark blue when SPEI-12 &ge; +1.0.
                </div>
                <div id="line-chart" class="chart-wrapper" style="visibility: hidden;"></div>
                <div id="bar-chart" class="chart-wrapper" style="visibility: hidden;"></div>
            </div>
            {PF_HTML}
        </div>
    </div>
    
    <div id="about-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="document.getElementById('about-modal').style.display='none'">&times;</span>
            <h2>About the Data Sources</h2>
            <div id="about-content"></div>
        </div>
    </div>

    <script>
        // Data injected from Python
        const geojsonData = {json.dumps(geojson_data)};
        const tsData = {json.dumps(ts_data)};
        const portfolioData = {json.dumps(portfolios, separators=(',', ':'))};
        const extraMetrics = {json.dumps(extra_metrics)};
        
        const metricNames = {{
            'hill1': 'Crop Diversity (Effective N)',
            'resp_div': 'Response Diversity',
            'stability': 'Stability (1/CV)',
            'resistance': 'Resistance (Median ratio)',
            'recovery': 'Recovery (Median ratio)',
            'vulnerability': 'Vulnerability (% years < 0.9)',
            'phi_sync': 'Synchrony'
        }};
        
        const sourcesMdContent = {json.dumps(sources_md_content)};
        document.getElementById('about-content').innerHTML = marked.parse(sourcesMdContent);
        
        const metricDescriptions = {{
            'hill1': 'The effective number of crops grown (exponent of Shannon entropy). Higher values indicate a more diverse crop portfolio.',
            'resp_div': 'Measures how differently a country\\'s crops react to drought. High values often flag extreme vulnerability in a single cash crop.',
            'stability': 'Inverse coefficient of variation (1/CV) of the national agricultural value. Higher means less year-to-year fluctuation.',
            'resistance': 'The ability to withstand a drought shock (ratio of production value during a drought year compared to pre-drought years).',
            'recovery': 'The ability to bounce back post-drought (ratio of post-drought production value compared to the drought year).',
            'vulnerability': 'The percentage of years where national agricultural value drops more than 10% below its long-term trend.',
            'phi_sync': 'Crop synchrony. Measures how correlated crop value fluctuations are. Lower synchrony generally improves stability.'
        }};
        
        Object.keys(extraMetrics).forEach(k => {{
            metricNames[k] = extraMetrics[k][0];
            metricDescriptions[k] = extraMetrics[k][1];
        }});

        let currentMetric = 'hill1';
        let currentCountry = null;
        
        // Colorscales (color-blind safe)
        const colorscales = {{
            'hill1': 'Viridis',
            'resp_div': 'Plasma',
            'stability': 'YlGnBu',
            'resistance': 'YlOrRd',
            'recovery': 'YlOrRd',
            'vulnerability': 'Reds',
            'phi_sync': 'Cividis'
        }};
        Object.keys(extraMetrics).forEach(k => {{ colorscales[k] = extraMetrics[k][2]; }});
        
        function renderMap() {{
            document.getElementById('metric-description').innerText = metricDescriptions[currentMetric];
            
            // Extract z values
            const zValues = [];
            const isoCodes = [];
            const hoverTexts = [];
            
            geojsonData.features.forEach(f => {{
                if (f.properties && f.properties[currentMetric] !== undefined) {{
                    zValues.push(f.properties[currentMetric]);
                    isoCodes.push(f.properties.iso3);
                    hoverTexts.push(f.properties.hover_text);
                }}
            }});
            
            const data = [{{
                type: 'choropleth',
                geojson: geojsonData,
                featureidkey: 'properties.iso3',
                locations: isoCodes,
                z: zValues,
                text: hoverTexts,
                hoverinfo: 'text',
                colorscale: colorscales[currentMetric],
                marker: {{ line: {{ color: '#fff', width: 0.5 }} }},
                colorbar: {{ 
                    title: metricNames[currentMetric], 
                    thickness: 10,
                    len: 0.8,
                    x: 0.98,
                    y: 0.5
                }}
            }}];
            
            const layout = {{
                geo: {{
                    showframe: false,
                    showcoastlines: true,
                    projection: {{ type: 'robinson' }},
                    coastlinecolor: '#ddd',
                    fitbounds: 'locations'
                }},
                margin: {{ t: 0, b: 0, l: 0, r: 0 }},
                clickmode: 'event+select'
            }};
            
            Plotly.newPlot('map', data, layout, {{responsive: true}});
            
            document.getElementById('map').on('plotly_click', function(data) {{
                if(data.points.length > 0) {{
                    const iso3 = data.points[0].location;
                    renderCharts(iso3);
                }}
            }});
        }}
        
        function renderCharts(iso3) {{
            if (!tsData[iso3]) return;
            
            currentCountry = iso3;
            document.getElementById('instruction').style.display = 'none';
            document.getElementById('chart-explanation').style.display = 'block';
            document.getElementById('line-chart').style.visibility = 'visible';
            document.getElementById('bar-chart').style.visibility = 'visible';
            
            const countryData = tsData[iso3];
            const years = countryData.years;
            
            // Build shapes for drought years
            const shapes = [];
            (countryData.wet_years || []).forEach(year => {{
                shapes.push({{
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: year - 0.5, x1: year + 0.5,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(33, 102, 172, 0.18)',
                    line: {{ width: 0 }}
                }});
            }});
            countryData.drought_years.forEach(year => {{
                shapes.push({{
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: year - 0.5, x1: year + 0.5,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(255, 0, 0, 0.15)',
                    line: {{ width: 0 }}
                }});
            }});
            
            // 1. Line Chart (Crop Anomalies)
            const lineData = [];
            const crops = Object.keys(countryData.crops);
            crops.forEach(crop => {{
                lineData.push({{
                    x: years,
                    y: countryData.crops[crop],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: crop,
                    line: {{ width: 2 }},
                    marker: {{ size: 4 }}
                }});
            }});
            
            const lineLayout = {{
                title: `${{iso3}} - Top 6 Crops Yield Anomalies`,
                margin: {{ t: 40, b: 20, l: 50, r: 20 }},
                xaxis: {{ title: 'Year', showgrid: true }},
                yaxis: {{ title: 'Yield Anomaly (relative)', zeroline: true, zerolinecolor: '#999' }},
                shapes: shapes,
                legend: {{ orientation: 'h', y: -0.2 }}
            }};
            
            Plotly.newPlot('line-chart', lineData, lineLayout, {{responsive: true}});
            
            // 2. Bar Chart (SPEI)
            const barColors = countryData.spei.map(v => v <= -1.0 ? '#e74c3c' : (v >= 1.0 ? '#1f4e9c' : '#aab7b8'));
            
            const barData = [{{
                x: years,
                y: countryData.spei,
                type: 'bar',
                marker: {{ color: barColors }},
                name: 'SPEI-12'
            }}];
            
            const barLayout = {{
                title: `${{iso3}} - Cropland-weighted SPEI-12`,
                margin: {{ t: 40, b: 40, l: 50, r: 20 }},
                xaxis: {{ title: 'Year' }},
                yaxis: {{ title: 'SPEI-12', zeroline: true, zerolinecolor: '#999' }},
                shapes: shapes
            }};
            
            Plotly.newPlot('bar-chart', barData, barLayout, {{responsive: true}});
            if (activeTab === 'pf') renderPortfolio();
        }}
        
        document.getElementById('metric-select').addEventListener('change', function(e) {{
            currentMetric = e.target.value;
            renderMap();
        }});
        
        {PF_JS_FILLED}

        // Initial render
        renderMap();
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('about-modal');
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}
    </script>
</body>
</html>
"""
    
    EXPLORER.mkdir(parents=True, exist_ok=True)
    out_file = EXPLORER / "index.html"
    if USE_DUMMY:
        (EXPLORER / "dummy").mkdir(exist_ok=True)
        out_file = EXPLORER / "dummy" / "index.html"
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Explorer built: {out_file} ({size_mb:.2f} MB)")
    print(f"Countries included: {len(included_iso3)}")
    print(f"Countries with a portfolio: {len(portfolios)}")
    print(f"Extra map layers: {list(extra_metrics) or 'none'}")
    assert size_mb < MAX_SIZE_MB, f"explorer is {size_mb:.2f} MB, limit {MAX_SIZE_MB} MB"

if __name__ == "__main__":
    build_explorer()
