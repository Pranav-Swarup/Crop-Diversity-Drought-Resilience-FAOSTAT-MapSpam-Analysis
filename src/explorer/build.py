import gzip
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.common import paths
from src.common.paths import (COUNTRIES_GEOJSON, METRICS, CROP_ANOMALIES, CLIMATE, EVENTS, EXPLORER,
                              USE_DUMMY, WET_EVENTS, METRICS_WET, TRADE)
from src.explorer.results import result_tables
from src.explorer.india_data import load_india_data

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


def _clean(v):
    """NaN -> None for JSON."""
    if v is None:
        return None
    try:
        return None if not np.isfinite(float(v)) else float(v)
    except (TypeError, ValueError):
        return v


def load_portfolios(valid_iso3):
    """Keep the exact arrays used by the optimizer and the published frontier."""
    if not PORTFOLIO_COV.exists():
        return {}
    portfolios = json.loads(PORTFOLIO_COV.read_text())["countries"]
    return {iso: data for iso, data in portfolios.items() if iso in valid_iso3}

# Plain (non f-string) blocks spliced into the template, so JS braces are written singly.
PF_CSS = """
        #tabs { display: flex; gap: 6px; padding: 8px 10px 0 10px; background: #fafafa; border-bottom: 1px solid #ccc; flex-wrap: wrap; }
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
        #india-filters { display: none; align-items: center; gap: 8px; flex-wrap: wrap; padding-top: 4px; border-top: 1px dashed #bdc3c7; }
        #india-filters select { padding: 6px 8px; font-size: 13px; border-radius: 4px; border: 1px solid #bdc3c7; max-width: 240px; }
        #india-jump-btn { display: none; background: #e67e22; color: #fff; border: none; padding: 7px 12px; border-radius: 4px; font-size: 13px; font-weight: bold; cursor: pointer; }
        #india-jump-btn:hover { background: #d35400; }
        #district-summary { display: none; background: #fff; border: 1px solid #dce1e5; border-left: 4px solid #2c3e50; border-radius: 4px; padding: 10px 12px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; }
        .ds-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 6px; margin-top: 6px; }
        .ds-item { background: #f8f9fa; padding: 4px 8px; border-radius: 3px; border: 1px solid #eee; }
        .ds-item span.lbl { color: #666; font-size: 11px; display: block; }
        .ds-item span.val { font-weight: bold; color: #2c3e50; font-size: 13px; }
        .ds-crops { margin-top: 6px; font-size: 12px; color: #444; }
        .ds-pill { display: inline-block; background: #eef4f7; border: 1px solid #d0dfe6; border-radius: 10px; padding: 1px 8px; margin: 2px 4px 2px 0; }
        #india-view { display: none; flex: 1; padding: 12px; overflow-y: auto; background: #fafafa; font-size: 14px; }
        .ind-card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
        .ind-card h3 { margin: 0 0 8px 0; font-size: 16px; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 6px; }
        .ind-note { font-size: 13px; color: #444; line-height: 1.5; margin-bottom: 10px; }
        .ind-table-wrap { max-height: 280px; overflow-y: auto; border: 1px solid #eee; margin-top: 8px; }
        .ind-table { border-collapse: collapse; width: 100%; font-size: 12.5px; background: #fff; }
        .ind-table th, .ind-table td { border: 1px solid #e0e0e0; padding: 5px 8px; text-align: right; }
        .ind-table th { background: #f4f6f7; position: sticky; top: 0; z-index: 2; color: #2c3e50; }
        .ind-table th:first-child, .ind-table td:first-child, .ind-table th:nth-child(2), .ind-table td:nth-child(2) { text-align: left; }
        .ind-plot { height: 320px; width: 100%; border: 1px solid #f0f0f0; border-radius: 4px; margin-top: 6px; }
        .ind-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; font-size: 13px; }
        .ind-toolbar select { padding: 5px 8px; font-size: 13px; border: 1px solid #bdc3c7; border-radius: 4px; }
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
        <div id="india-view">
            <div class="ind-card">
                <h3>Phase 4: Crop diversity and drought outcomes within Indian districts</h3>
                <div class="ind-note">
                    The national panel uses crop diversity measured over the full analysis window.
                    Phase 4 estimates the association <strong>within Indian districts over time</strong> using 5-year lagged crop diversity (<code>hill1_tv</code>, years <em>t&minus;5..t&minus;1</em>)
                    with district (<code>&alpha;<sub>d</sub></code>) and state&times;year (<code>&lambda;<sub>s,t</sub></code>) fixed effects:<br>
                    <code>anom<sub>d,t</sub> = &beta;&middot;spei<sub>d,t</sub> + &gamma;&middot;spei<sub>d,t</sub> &times; z(hill1_tv<sub>d,t</sub>) + &delta;&middot;z(hill1_tv<sub>d,t</sub>) + &alpha;<sub>d</sub> + &lambda;<sub>state&times;year</sub> + &varepsilon;<sub>d,t</sub></code>
                </div>
                <div id="ind-summary-wrap"></div>
            </div>

            <div class="ind-card">
                <h3>1. District Panel Regressions &amp; Irrigation Confounder Test</h3>
                <div class="ind-toolbar">
                    <label for="ind-panel-win"><strong>Window / Sample:</strong></label>
                    <select id="ind-panel-win">
                        <option value="core">Core DES (1998&ndash;2022)</option>
                        <option value="core_allstable">All Stable DES (1998&ndash;2022, incl. split-flagged)</option>
                        <option value="ext5">Extended 5-Crop Mirror (1993&ndash;2015)</option>
                        <option value="ext5_allstable">All Stable Extended 5-Crop (1993&ndash;2015)</option>
                    </select>
                </div>
                <div id="ind-panel-chart" class="ind-plot"></div>
                <div class="ind-table-wrap"><table class="ind-table" id="ind-panel-table"></table></div>
            </div>

            <div class="ind-card">
                <h3>2. Drought Severity Bins &amp; Within-State Studentised Permutation</h3>
                <div class="ind-note" id="ind-perm-note"></div>
                <div id="ind-sev-chart" class="ind-plot"></div>
                <div class="ind-table-wrap"><table class="ind-table" id="ind-sev-table"></table></div>
            </div>

            <div class="ind-card">
                <h3>3. Four Faces of Resilience: Indian Districts vs. Global National Cross-Section</h3>
                <div class="ind-toolbar">
                    <label for="ind-face-pred"><strong>Predictor:</strong></label>
                    <select id="ind-face-pred">
                        <option value="hill1">Crop Diversity (hill1)</option>
                        <option value="resp_div">Response Diversity (resp_div)</option>
                        <option value="resp_divergence">Response Divergence (resp_divergence)</option>
                    </select>
                </div>
                <div id="ind-faces-chart" class="ind-plot"></div>
                <div class="ind-table-wrap"><table class="ind-table" id="ind-faces-table"></table></div>
            </div>

            <div class="ind-card">
                <h3>4. Robustness: Spatial-HAC (Conley), Kharif Monsoon Exposure &amp; Crops, Leave-One-State-Out</h3>
                <div class="ind-toolbar">
                    <label for="ind-rob-view"><strong>Robustness Check:</strong></label>
                    <select id="ind-rob-view">
                        <option value="conley">Conley Spatial-HAC Standard Errors (100, 200, 400 km)</option>
                        <option value="kharif">Kharif Monsoon SPEI-4 (Sept) &amp; Kharif-Only Crop Tables</option>
                        <option value="loso">Leave-One-State-Out Sensitivity (spei &times; z(hill1_tv))</option>
                    </select>
                </div>
                <div id="ind-rob-chart" class="ind-plot"></div>
                <div class="ind-table-wrap"><table class="ind-table" id="ind-rob-table"></table></div>
            </div>

            <div class="ind-card">
                <h3>5. Ground-Truth Validation: Documented All-India Monsoon Drought Years</h3>
                <div class="ind-note">
                    District-level drought incidence (share of districts with <code>spei12_w &le; &minus;1.0</code>) against India's documented monsoon drought years with published rainfall departures.
                </div>
                <div id="ind-val-chart" class="ind-plot"></div>
                <div class="ind-table-wrap"><table class="ind-table" id="ind-val-table"></table></div>
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
        let indiaResultsRendered = false;

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
            const empty = document.getElementById('pf-empty');
            const body = document.getElementById('pf-body');
            if (currentScope !== 'global') {
                body.style.display = 'none';
                empty.style.display = 'block';
                empty.innerHTML = 'The country-level portfolio simulator uses national FAOSTAT covariance matrices (1993&ndash;2023). ' +
                    'Switch <strong>Map Scope</strong> to <em>Global</em> and click India (<code>IND</code>) or another country to simulate national crop mix reallocations, or view the <strong>Time series</strong> or <strong>Phase 4: India Results</strong> tabs.';
                pf = null;
                return;
            }
            if (!currentCountry) return;
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

        // ---------------- Phase 4 India Statistical Results Tab ----------------
        function fmtNum(v, d=4) {
            if (v === null || v === undefined || Number.isNaN(v)) return '&mdash;';
            const n = Number(v);
            if (Math.abs(n) > 0 && Math.abs(n) < 10 ** (-d)) return n.toExponential(2);
            return n.toFixed(d);
        }

        function renderIndiaSummary() {
            if (!indiaData || !indiaData.stats) return;
            const sRows = indiaData.stats.summary || [];
            const pRows = indiaData.stats.panel || [];
            const getBeta = (m, t) => {
                const r = pRows.find(x => x.model === m && x.term === t);
                return r ? `${fmtNum(r.beta, 4)} (p=${fmtNum(r.p, 3)})` : '&mdash;';
            };
            let html = '<table class="ind-table"><thead><tr>' +
                '<th>Window</th><th>Years</th><th>Usable Districts</th><th>Drought Events (Dist w/ Event)</th>' +
                '<th>Median Hill-1</th><th>Median Value Coverage</th><th>Core &beta;<sub>SPEI</sub></th>' +
                '<th>Core &gamma;<sub>SPEI&times;z(hill1_tv)</sub></th><th>Irrigation &beta;<sub>SPEI&times;z(irrig_lag)</sub></th>' +
                '</tr></thead><tbody>';
            sRows.forEach(r => {
                const w = r.window;
                html += `<tr>` +
                    `<td><strong>${w === 'core' ? 'Core DES (47 crops)' : 'Extended Mirror (5 crops)'}</strong></td>` +
                    `<td>${r.years}</td>` +
                    `<td>${r.districts}</td>` +
                    `<td>${r.n_events} (${r.districts_with_event})</td>` +
                    `<td>${fmtNum(r.median_hill1, 2)}</td>` +
                    `<td>${percent(r.median_value_coverage)}</td>` +
                    `<td>${getBeta(w + ':core', 'spei')}</td>` +
                    `<td>${getBeta(w + ':core', 'spei_z')}</td>` +
                    `<td>${getBeta(w + ':le2015_irrig', 'spei_x_irrig_lag')}</td>` +
                    `</tr>`;
            });
            html += '</tbody></table>';
            document.getElementById('ind-summary-wrap').innerHTML = html;
        }

        function renderIndiaPanelSection() {
            if (!indiaData || !indiaData.stats) return;
            const winPrefix = document.getElementById('ind-panel-win').value;
            const rows = (indiaData.stats.panel || []).filter(r => r.model.startsWith(winPrefix + ':'));
            const focusRows = rows.filter(r => ['spei', 'spei_z', 'spei_z3', 'dry', 'dry_z', 'spei_x_irrig_lag', 'spei_x_fert_lag'].includes(r.term));
            const yLabels = focusRows.map(r => `${r.model.replace(winPrefix + ':', '')} [${r.term}]`).reverse();
            const betas = focusRows.map(r => r.beta).reverse();
            const errHi = focusRows.map(r => r.ci_hi - r.beta).reverse();
            const errLo = focusRows.map(r => r.beta - r.ci_lo).reverse();
            const colors = focusRows.map(r => r.term.includes('irrig') ? '#2980b9' : (r.term.includes('z') ? '#e67e22' : '#2c3e50')).reverse();

            Plotly.newPlot('ind-panel-chart', [{
                x: betas,
                y: yLabels,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 9, color: colors },
                error_x: { type: 'data', symmetric: false, array: errHi, arrayminus: errLo, color: '#555', thickness: 1.5, width: 4 },
                hovertemplate: '<b>%{y}</b><br>&beta; = %{x:.4f}<extra></extra>'
            }], {
                title: { text: `Panel Coefficients &amp; 95% CIs (${winPrefix})`, font: { size: 14 } },
                margin: { t: 35, b: 40, l: 175, r: 20 },
                xaxis: { title: 'Coefficient (&beta;, share of trend per unit)', zeroline: true, zerolinecolor: '#c0392b', zerolinewidth: 1.5 },
                yaxis: { automargin: true }
            }, { responsive: true });

            let thtml = '<thead><tr><th>Model</th><th>Term</th><th>&beta;</th><th>SE</th><th>95% CI</th><th>p-value</th><th>Obs (Districts)</th><th>R&sup2; within</th><th>Note</th></tr></thead><tbody>';
            rows.forEach(r => {
                thtml += `<tr><td>${r.model}</td><td><code>${r.term}</code></td><td>${fmtNum(r.beta, 4)}</td><td>${fmtNum(r.se, 4)}</td>` +
                    `<td>[${fmtNum(r.ci_lo, 4)}, ${fmtNum(r.ci_hi, 4)}]</td><td>${fmtNum(r.p, 4)}</td>` +
                    `<td>${r.n_obs} (${r.n_districts})</td><td>${fmtNum(r.r2_within, 4)}</td><td style="text-align:left;">${r.note || ''}</td></tr>`;
            });
            thtml += '</tbody>';
            document.getElementById('ind-panel-table').innerHTML = thtml;
        }

        function renderIndiaSeveritySection() {
            if (!indiaData || !indiaData.stats) return;
            const sev = indiaData.stats.severity || [];
            const cnts = indiaData.stats.severity_counts || [];
            const perms = indiaData.stats.severity_perm || [];

            const permText = perms.map(p =>
                `<strong>${p.model}</strong> extreme &times; diversity (<code>${p.term}</code>): <em>t</em><sub>obs</sub> = ${fmtNum(p.t_obs, 2)}, ` +
                `studentised within-state permutation <strong>p<sub>perm_t</sub> = ${fmtNum(p.p_perm_t, 3)}</strong> (${p.n_perm} permutations)`
            ).join(' &nbsp;|&nbsp; ');
            document.getElementById('ind-perm-note').innerHTML =
                `Drought severity bins: <code>moderate</code> (&minus;1.5 &lt; SPEI &le; &minus;1.0), <code>severe</code> (&minus;2.0 &lt; SPEI &le; &minus;1.5), <code>extreme</code> (SPEI &le; &minus;2.0).<br>` +
                permText;

            const bins = ['moderate_z', 'severe_z', 'extreme_z'];
            const binLabels = ['Moderate &times; z(hill1_tv)', 'Severe &times; z(hill1_tv)', 'Extreme &times; z(hill1_tv)'];
            const traces = [];
            [['core:severity', 'Core DES', '#2980b9'],
             ['core_allstable:severity', 'All Stable DES', '#5dade2'],
             ['ext5:severity', 'Extended 5-Crop', '#d35400'],
             ['ext5_allstable:severity', 'All Stable 5-Crop', '#f39c12']].forEach(([mName, label, colr]) => {
                const sub = bins.map(b => sev.find(r => r.model === mName && r.term === b)).filter(Boolean);
                if (sub.length === 3) {
                    traces.push({
                        x: binLabels,
                        y: sub.map(r => r.beta),
                        name: label,
                        type: 'bar',
                        marker: { color: colr },
                        error_y: {
                            type: 'data',
                            symmetric: false,
                            array: sub.map(r => r.ci_hi - r.beta),
                            arrayminus: sub.map(r => r.beta - r.ci_lo)
                        }
                    });
                }
            });

            Plotly.newPlot('ind-sev-chart', traces, {
                barmode: 'group',
                title: { text: 'Diversity &times; Drought Severity Bin Coefficients (&plusmn; 95% CI)', font: { size: 14 } },
                margin: { t: 35, b: 45, l: 60, r: 20 },
                yaxis: { title: 'Change in anomaly per SD of diversity', zeroline: true, zerolinecolor: '#666' },
                legend: { orientation: 'h', y: -0.22 }
            }, { responsive: true });

            const cntMap = {};
            cnts.forEach(c => { cntMap[c.model + ':' + c.bin] = `${c.n_district_years} (top tercile: ${c.n_high_diversity_tercile})`; });
            let thtml = '<thead><tr><th>Model</th><th>Term</th><th>&beta;</th><th>SE</th><th>95% CI</th><th>p-value</th><th>Bin Dist-Years (High Div Tercile)</th></tr></thead><tbody>';
            sev.forEach(r => {
                const w = r.model.split(':')[0];
                const baseBin = r.term.replace('_z', '');
                const cInfo = cntMap[w + ':' + baseBin] || '&mdash;';
                thtml += `<tr><td>${r.model}</td><td><code>${r.term}</code></td><td>${fmtNum(r.beta, 4)}</td><td>${fmtNum(r.se, 4)}</td>` +
                    `<td>[${fmtNum(r.ci_lo, 4)}, ${fmtNum(r.ci_hi, 4)}]</td><td>${fmtNum(r.p, 4)}</td><td>${cInfo}</td></tr>`;
            });
            thtml += '</tbody>';
            document.getElementById('ind-sev-table').innerHTML = thtml;
        }

        function renderIndiaFacesSection() {
            if (!indiaData || !indiaData.stats) return;
            const pred = document.getElementById('ind-face-pred').value;
            const rows = (indiaData.stats.vs_national || []).filter(r => r.predictor === pred);
            const coreRows = rows.filter(r => r.window === 'core');
            const ext5Rows = rows.filter(r => r.window === 'ext5');
            const faces = coreRows.map(r => r.face);

            const traces = [
                {
                    x: faces,
                    y: coreRows.map(r => r.rho),
                    name: 'India Districts — Core DES',
                    customdata: coreRows.map(r => r.n),
                    hovertemplate: '%{x}: %{y:.4f} (n=%{customdata})<extra>%{fullData.name}</extra>',
                    type: 'bar',
                    marker: { color: '#2980b9' },
                    error_y: { type: 'data', symmetric: false, array: coreRows.map(r => r.ci_hi - r.rho), arrayminus: coreRows.map(r => r.rho - r.ci_lo) }
                },
                {
                    x: faces,
                    y: ext5Rows.map(r => r.rho),
                    name: 'India Districts — Extended 5-Crop',
                    customdata: ext5Rows.map(r => r.n),
                    hovertemplate: '%{x}: %{y:.4f} (n=%{customdata})<extra>%{fullData.name}</extra>',
                    type: 'bar',
                    marker: { color: '#e67e22' },
                    error_y: { type: 'data', symmetric: false, array: ext5Rows.map(r => r.ci_hi - r.rho), arrayminus: ext5Rows.map(r => r.rho - r.ci_lo) }
                }
            ];
            if (coreRows.some(r => r.rho_national !== null)) {
                traces.push({
                    x: faces,
                    y: coreRows.map(r => r.rho_national),
                    name: 'Global National Cross-Section',
                    customdata: coreRows.map(r => r.n_national),
                    hovertemplate: '%{x}: %{y:.4f} (n=%{customdata})<extra>%{fullData.name}</extra>',
                    type: 'bar',
                    marker: { color: '#27ae60' }
                });
            }

            Plotly.newPlot('ind-faces-chart', traces, {
                barmode: 'group',
                title: { text: `Spearman Rank Correlation (&rho;) of ${pred} vs. Four Faces of Resilience`, font: { size: 14 } },
                margin: { t: 35, b: 45, l: 60, r: 20 },
                yaxis: { title: 'Spearman &rho;', range: [-1, 1], zeroline: true, zerolinecolor: '#666' },
                legend: { orientation: 'h', y: -0.22 }
            }, { responsive: true });

            let thtml = '<thead><tr><th>Window</th><th>Face</th><th>Predictor</th><th>India &rho;</th><th>India 95% CI</th><th>India p</th><th>India n</th><th>National &rho;</th><th>National p</th><th>National n</th></tr></thead><tbody>';
            rows.forEach(r => {
                thtml += `<tr><td>${r.window}</td><td><strong>${r.face}</strong></td><td><code>${r.predictor}</code></td>` +
                    `<td>${fmtNum(r.rho, 3)}</td><td>[${fmtNum(r.ci_lo, 3)}, ${fmtNum(r.ci_hi, 3)}]</td><td>${fmtNum(r.p, 4)}</td><td>${r.n}</td>` +
                    `<td>${fmtNum(r.rho_national, 3)}</td><td>${fmtNum(r.p_national, 4)}</td><td>${r.n_national !== null ? r.n_national : '&mdash;'}</td></tr>`;
            });
            thtml += '</tbody>';
            document.getElementById('ind-faces-table').innerHTML = thtml;
        }

        function renderIndiaRobustnessSection() {
            if (!indiaData || !indiaData.stats) return;
            const mode = document.getElementById('ind-rob-view').value;
            if (mode === 'conley') {
                const rows = indiaData.stats.conley || [];
                const speiZ = rows.filter(r => r.term === 'spei_z');
                const spei = rows.filter(r => r.term === 'spei');
                Plotly.newPlot('ind-rob-chart', [
                    {
                        x: spei.map(r => r.model),
                        y: spei.map(r => r.se),
                        name: 'SE of spei',
                        type: 'bar',
                        marker: { color: '#2c3e50' }
                    },
                    {
                        x: speiZ.map(r => r.model),
                        y: speiZ.map(r => r.se),
                        name: 'SE of spei &times; z(hill1_tv)',
                        type: 'bar',
                        marker: { color: '#e67e22' }
                    }
                ], {
                    barmode: 'group',
                    title: { text: 'Standard Error Comparison: District CR1, State CR1, and Conley Spatial-HAC (100, 200, 400 km)', font: { size: 14 } },
                    margin: { t: 35, b: 65, l: 60, r: 20 },
                    yaxis: { title: 'Standard Error (SE)' },
                    legend: { orientation: 'h', y: -0.3 }
                }, { responsive: true });

                let thtml = '<thead><tr><th>Model</th><th>Term</th><th>&beta;</th><th>SE</th><th>95% CI</th><th>p-value</th><th>Clusters / Districts</th><th>Note</th></tr></thead><tbody>';
                rows.forEach(r => {
                    thtml += `<tr><td>${r.model}</td><td><code>${r.term}</code></td><td>${fmtNum(r.beta, 4)}</td><td>${fmtNum(r.se, 4)}</td>` +
                        `<td>[${fmtNum(r.ci_lo, 4)}, ${fmtNum(r.ci_hi, 4)}]</td><td>${fmtNum(r.p, 4)}</td><td>${r.n_clusters ?? '—'} / ${r.n_districts}</td><td style="text-align:left;">${r.note || ''}</td></tr>`;
                });
                thtml += '</tbody>';
                document.getElementById('ind-rob-table').innerHTML = thtml;
            } else if (mode === 'kharif') {
                const kh = (indiaData.stats.kharif || []).concat(indiaData.stats.kharif_crops || []);
                const sub = kh.filter(r => ['spei','spei_z','extreme_z'].includes(r.term)).reverse();
                Plotly.newPlot('ind-rob-chart', [{
                    x: sub.map(r => r.beta),
                    y: sub.map(r => `${r.model} [${r.term}]`),
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 8, color: '#16a085' },
                    error_x: {
                        type: 'data',
                        symmetric: false,
                        array: sub.map(r => (r.ci_hi !== null ? r.ci_hi - r.beta : 0)),
                        arrayminus: sub.map(r => (r.ci_lo !== null ? r.beta - r.ci_lo : 0))
                    }
                }], {
                    title: { text: 'Kharif Exposure (Sept SPEI-4) &amp; Kharif-Only Crop Models (&beta; &plusmn; 95% CI)', font: { size: 14 } },
                    margin: { t: 35, b: 40, l: 220, r: 20 },
                    xaxis: { title: 'Coefficient (&beta;)', zeroline: true, zerolinecolor: '#c0392b' }
                }, { responsive: true });

                let thtml = '<thead><tr><th>Model</th><th>Term</th><th>&beta; / &rho;</th><th>SE</th><th>95% CI</th><th>p-value</th><th>Obs (Districts)</th></tr></thead><tbody>';
                kh.forEach(r => {
                    thtml += `<tr><td>${r.model}</td><td><code>${r.term}</code></td><td>${fmtNum(r.beta, 4)}</td><td>${fmtNum(r.se, 4)}</td>` +
                        `<td>${r.ci_lo !== null ? '[' + fmtNum(r.ci_lo, 4) + ', ' + fmtNum(r.ci_hi, 4) + ']' : '&mdash;'}</td>` +
                        `<td>${fmtNum(r.p, 4)}</td><td>${r.n_obs}${r.n_districts ? ' (' + r.n_districts + ')' : ''}</td></tr>`;
                });
                thtml += '</tbody>';
                const permutations = [...indiaData.stats.kharif_perm, ...indiaData.stats.kharif_crops_perm];
                thtml += '<tbody><tr><th>Permutation model</th><th>Term</th><th>Observed t</th><th>Permutation p</th><th>Draws</th></tr>';
                for (const row of permutations) {
                    thtml += `<tr><td>${row.model}</td><td>${row.term}</td><td>${fmtNum(row.t_obs)}</td><td>${row.p_perm_t}</td><td>${row.n_perm}</td></tr>`;
                }
                thtml += '</tbody>';
                document.getElementById('ind-rob-table').innerHTML = thtml;
            } else {
                const loso = (indiaData.stats.leave_out_state || []).filter(r => r.model === 'core:leave_out_state');
                Plotly.newPlot('ind-rob-chart', [{
                    x: loso.map(r => r.dropped_state),
                    y: loso.map(r => r.beta_spei_z),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'spei &times; z(hill1_tv)',
                    marker: { size: 8, color: '#8e44ad' },
                    error_y: { type: 'data', array: loso.map(r => 1.96 * r.se), color: '#8e44ad' }
                }], {
                    title: { text: 'Leave-One-State-Out Stability of spei &times; z(hill1_tv) (Core DES, &plusmn;1.96 SE)', font: { size: 14 } },
                    margin: { t: 35, b: 85, l: 60, r: 20 },
                    xaxis: { tickangle: -35 },
                    yaxis: { title: '&beta; (spei &times; z(hill1_tv))', zeroline: true, zerolinecolor: '#666' }
                }, { responsive: true });

                let thtml = '<thead><tr><th>Model</th><th>Dropped State</th><th>&beta; (spei_z)</th><th>SE</th><th>p-value</th><th>Obs</th><th>Districts</th></tr></thead><tbody>';
                (indiaData.stats.leave_out_state || []).forEach(r => {
                    thtml += `<tr><td>${r.model}</td><td><strong>${r.dropped_state}</strong></td><td>${fmtNum(r.beta_spei_z, 4)}</td>` +
                        `<td>${fmtNum(r.se, 4)}</td><td>${fmtNum(r.p, 4)}</td><td>${r.n_obs}</td><td>${r.n_districts}</td></tr>`;
                });
                thtml += '</tbody>';
                document.getElementById('ind-rob-table').innerHTML = thtml;
            }
        }

        function renderIndiaValidationSection() {
            if (!indiaData || !indiaData.stats) return;
            const rows = indiaData.stats.drought_years || [];
            const yrs = rows.map(r => r.year);
            const sharePct = rows.map(r => r.share_drought == null ? null : 100 * r.share_drought);
            const barCols = rows.map(r => r.documented_drought_year ? '#c0392b' : '#95a5a6');

            Plotly.newPlot('ind-val-chart', [
                {
                    x: yrs,
                    y: sharePct,
                    type: 'bar',
                    name: 'Districts in Drought (SPEI-12 <= -1.0, %)',
                    marker: { color: barCols }
                },
                {
                    x: yrs,
                    y: rows.map(r => r.mean_spei),
                    yaxis: 'y2',
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'All-India Mean District SPEI-12',
                    line: { color: '#2c3e50', width: 2 }
                }
            ], {
                title: { text: 'Share of Indian Districts in Drought vs. Documented Monsoon Drought Years (Red Bars)', font: { size: 14 } },
                margin: { t: 35, b: 45, l: 55, r: 55 },
                yaxis: { title: 'Districts with SPEI-12 <= -1.0 (%)' },
                yaxis2: { title: 'Mean SPEI-12', overlaying: 'y', side: 'right', showgrid: false, zeroline: true },
                legend: { orientation: 'h', y: -0.22 }
            }, { responsive: true });

            let thtml = '<thead><tr><th>Year</th><th>Documented Drought</th><th>Districts in Drought (%)</th><th>Mean District SPEI-12</th><th>Monsoon Departure (% LPA)</th><th>Citation / Source</th></tr></thead><tbody>';
            rows.forEach(r => {
                const srcHtml = r.source ? (r.url ? `<a href="${r.url}" target="_blank" rel="noopener">${r.source}</a>` : r.source) : '&mdash;';
                thtml += `<tr style="${r.documented_drought_year ? 'background:#fdf2e9;font-weight:bold;' : ''}">` +
                    `<td>${r.year}</td><td>${r.documented_drought_year ? 'Documented' : 'Unclassified'}</td>` +
                    `<td>${percent(r.share_drought)}</td><td>${fmtNum(r.mean_spei, 2)}</td>` +
                    `<td>${r.monsoon_departure_lpa_pct !== null ? fmtNum(r.monsoon_departure_lpa_pct, 0) + '%' : '&mdash;'}</td>` +
                    `<td style="text-align:left;font-weight:normal;">${srcHtml}</td></tr>`;
            });
            thtml += '</tbody>';
            document.getElementById('ind-val-table').innerHTML = thtml;
        }

        function renderIndiaResultsTab() {
            if (!indiaData || !indiaData.stats) return;
            renderIndiaSummary();
            renderIndiaPanelSection();
            renderIndiaSeveritySection();
            renderIndiaFacesSection();
            renderIndiaRobustnessSection();
            renderIndiaValidationSection();
            indiaResultsRendered = true;
        }

        document.getElementById('ind-panel-win').addEventListener('change', renderIndiaPanelSection);
        document.getElementById('ind-face-pred').addEventListener('change', renderIndiaFacesSection);
        document.getElementById('ind-rob-view').addEventListener('change', renderIndiaRobustnessSection);

        function showTab(name) {
            activeTab = name;
            document.getElementById('tab-national').classList.toggle('active', name === 'national');
            document.getElementById('national-view').style.display = name === 'national' ? 'block' : 'none';
            document.getElementById('tab-ts').classList.toggle('active', name === 'ts');
            document.getElementById('tab-pf').classList.toggle('active', name === 'pf');
            const tabInd = document.getElementById('tab-india');
            if (tabInd) tabInd.classList.toggle('active', name === 'india');
            document.getElementById('charts-inner').style.display = name === 'ts' ? 'flex' : 'none';
            document.getElementById('pf-view').style.display = name === 'pf' ? 'block' : 'none';
            document.getElementById('india-view').style.display = name === 'india' ? 'block' : 'none';
            if (name === 'pf') {
                renderPortfolio();
            } else if (name === 'india') {
                if (!indiaResultsRendered) renderIndiaResultsTab();
                else {
                    ['ind-panel-chart', 'ind-sev-chart', 'ind-faces-chart', 'ind-rob-chart', 'ind-val-chart'].forEach(id => {
                        Plotly.Plots.resize(id);
                    });
                }
            } else if (name === 'ts' && (currentCountry || currentDistrict)) {
                Plotly.Plots.resize('line-chart');
                Plotly.Plots.resize('bar-chart');
            }
        }
        const tableSelect = document.getElementById('national-select');
        const pageSize = 50;
        let page = 0;
        for (const key of Object.keys(nationalTables)) tableSelect.add(new Option(key.replaceAll('_',' '), key));
        function renderNational() {
            const {columns, rows} = nationalTables[tableSelect.value];
            const t = document.getElementById('national-table');
            t.replaceChildren();
            const header = t.createTHead().insertRow();
            for (const name of columns) header.appendChild(document.createElement('th')).textContent = name;
            const body = t.createTBody();
            for (const row of rows.slice(page * pageSize, (page+1) * pageSize)) {
                const tr = body.insertRow();
                for (const value of row) tr.insertCell().textContent = value ?? 'n/a';
            }
            document.getElementById('national-count').textContent = `${rows.length} rows · page ${page+1} of ${Math.max(1,Math.ceil(rows.length/pageSize))}`;
            document.getElementById('national-prev').disabled = page === 0;
            document.getElementById('national-next').disabled = (page+1)*pageSize >= rows.length;
        }
        tableSelect.onchange = () => { page = 0; renderNational(); };
        document.getElementById('national-prev').onclick = () => { page--; renderNational(); };
        document.getElementById('national-next').onclick = () => { page++; renderNational(); };
        document.getElementById('tab-national').onclick = () => { showTab('national'); renderNational(); };
        document.getElementById('tab-ts').addEventListener('click', () => showTab('ts'));
        document.getElementById('tab-pf').addEventListener('click', () => showTab('pf'));
        if (document.getElementById('tab-india')) {
            document.getElementById('tab-india').addEventListener('click', () => showTab('india'));
            if (!indiaData) document.getElementById('tab-india').style.display = 'none';
        }
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
    
    assets = Path(__file__).with_name("assets")
    plotly_js = gzip.decompress((assets / "plotly-2.32.0.min.js.gz").read_bytes()).decode()
    notices = "\n".join((assets / name).read_text() for name in
                        ("plotly-LICENSE.txt", "plotly.min.js.LICENSE.txt"))
    topology = (assets / "world_110m.json").read_text()
    sources = ("<p>This synthetic demonstration uses generated crop and climate fixtures. "
               "Its country names are real and its numbers are simulated. The observational "
               "results and their sources are in the main explorer.</p>" if USE_DUMMY
               else (assets / "sources.html").read_text())
    tables = result_tables()

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
            feature['properties'].update({k: _clean(v) for k, v in metrics_dict[iso3].items()})
            for key, vals in extra_values.items():
                feature['properties'][key] = vals.get(iso3)
            
    # 4. Prepare timeseries data for the right panel
    print("  Preparing timeseries data...")
    ts_data = {}
    for iso3 in included_iso3:
        # Climate
        c_df = climate_df[climate_df['iso3'] == iso3].sort_values('year')
        years = c_df['year'].tolist()
        spei = [_clean(v) for v in c_df['spei12_w'].tolist()]
        
        # Events
        e_df = events_df[events_df['iso3'] == iso3]
        drought_years = e_df['year'].tolist()
        wet_years = wet_df[wet_df['iso3'] == iso3]['year'].tolist()
        
        # Top 6 crops by mean value_share
        a_df = anom_df[anom_df['iso3'] == iso3]
            
        crop_shares = a_df.groupby('item')['value_share'].first().sort_values(ascending=False)
        top_crops = crop_shares.head(6).index.tolist()
        
        crops_data = {}
        for crop in top_crops:
            crop_df = a_df[a_df['item'] == crop].set_index('year')
            # align with global years
            aligned = crop_df.reindex(years)['yield_anom'].tolist()
            crops_data[crop] = [_clean(v) for v in aligned]
            
        ts_data[iso3] = {
            'years': years,
            'spei': spei,
            'drought_years': drought_years,
            'wet_years': wet_years,
            'crops': crops_data
        }

    portfolios = load_portfolios(set(ts_data))
    print(f"  Portfolios embedded: {len(portfolios)}")
    india_data = load_india_data()
    extra_options = "".join(f'<option value="{k}" data-scope="global">{v[0]}</option>' for k, v in extra_metrics.items())
    india_extra_options = ""
    if india_data:
        india_extra_options = "".join(
            f'<option value="{k}" data-scope="india" style="display:none;">{v[0]}</option>'
            for k, v in india_data["extra_metrics"].items()
        )

    # 5. HTML Template
    print("  Generating HTML...")
    
    PF_JS_FILLED = PF_JS.replace("__MAX_SLIDERS__", str(MAX_SLIDERS))
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Crop Diversity & Drought Resilience Explorer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script type="text/plain" id="plotly-licenses">{notices}</script>
    <script>{plotly_js}</script>
    <script>window.PlotlyGeoAssets = {{topojson: {{world_110m: {topology}}}}};</script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }}
        #header {{ background: #2c3e50; color: white; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }}
        #header h1 {{ margin: 0; font-size: 20px; }}
        #header button {{ background: #34495e; color: white; border: 1px solid #7f8c8d; padding: 8px 15px; cursor: pointer; border-radius: 4px; }}
        #header button:hover {{ background: #7f8c8d; }}
        #main {{ display: flex; flex: 1; overflow: hidden; }}
        #map-container {{ flex: 1; border-right: 1px solid #ccc; display: flex; flex-direction: column; }}
        #controls-wrapper {{ padding: 15px; background: #ecf0f1; border-bottom: 1px solid #bdc3c7; display: flex; flex-direction: column; gap: 10px; }}
        #controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
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
        #map-container, #charts-container, #controls, #charts-inner {{ min-width:0; }}
        @media (max-width: 820px) {{
            body {{ height:auto; }}
            #main {{ flex-direction:column; overflow:visible; }}
            #header, #header > div, #tabs, #india-filters {{ flex-wrap:wrap; gap:6px; }}
            #header {{ align-items:flex-start; }}
            #map-container, #charts-container {{ flex:auto; width:100%; box-sizing:border-box; overflow:visible; }}
            #map {{ flex:none; height:52vh; min-height:260px; }}
            #charts-inner {{ min-height:350px; }}
            #controls select, #india-filters select {{ max-width:100%; }}
            .ind-card, .ind-table-wrap {{ min-width:0; overflow:auto; }}
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Crop Diversity & Drought Resilience Explorer{' — Synthetic demonstration' if USE_DUMMY else ''}</h1>
        <div style="display:flex; gap:10px; align-items:center;">
            <button id="india-jump-btn" onclick="switchScope('india_core')">Explore India Districts (Phase 4) &rarr;</button>
            <button onclick="document.getElementById('about-modal').style.display='block'">About the Data</button>
        </div>
    </div>
    
    <div id="main">
        <div id="map-container">
            <div id="controls-wrapper">
                <div id="controls">
                    <label for="scope-select" id="scope-label"><strong>Map Scope:</strong></label>
                    <select id="scope-select">
                        <option value="global">Global ({len(included_iso3)} Countries, 1993&ndash;2023)</option>
                        <option value="india_core">India Districts &mdash; Core DES ({len(india_data["metrics"]["core"]) if india_data else 0} Districts, 1998&ndash;2022)</option>
                        <option value="india_ext5">India Districts &mdash; Extended 5-Crop ({len(india_data["metrics"]["ext5"]) if india_data else 0} Districts, 1993&ndash;2015)</option>
                    </select>
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
                        {india_extra_options}
                    </select>
                </div>
                <div id="india-filters">
                    <label for="india-state-select"><strong>State:</strong></label>
                    <select id="india-state-select"><option value="ALL">All States</option></select>
                    <label for="india-dist-select"><strong>District:</strong></label>
                    <select id="india-dist-select"><option value="">Select a district...</option></select>
                </div>
                <div id="metric-description"></div>
            </div>
            <div id="map"></div>
        </div>
        
        <div id="charts-container">
            <div id="tabs">
                <button id="tab-ts" class="tab-btn active">Time series</button>
                <button id="tab-pf" class="tab-btn">Portfolio simulator</button>
                <button id="tab-national" class="tab-btn">Data tables</button>
                <button id="tab-india" class="tab-btn">Phase 4: India District Results</button>
            </div>
            <div id="charts-inner">
                <div id="instruction" class="instruction-overlay">Click a country on the map to view timeseries</div>
                <div id="district-summary"></div>
                <div id="chart-explanation">
                    <strong>Top Chart:</strong> Shows the yield anomalies (deviations from long-term trends) for the country's top 6 crops by value.<br>
                    <strong>Bottom Chart:</strong> Shows the Cropland-weighted SPEI-12 climate index. Negative values indicate drier conditions.<br>
                    <strong>Red Shaded Regions:</strong> Mark drought events in the calculated catalogue (SPEI-12 &le; -1.0) based on the project's criteria.<br>
                    <strong>Blue Shaded Regions:</strong> Represent wet events (SPEI-12 &ge; +1.0), the mirror of the drought rule. Bars are red when SPEI-12 &le; -1.0 and dark blue when SPEI-12 &ge; +1.0.
                </div>
                <div id="line-chart" class="chart-wrapper" style="visibility: hidden;"></div>
                <div id="bar-chart" class="chart-wrapper" style="visibility: hidden;"></div>
            </div>
            <section id="national-view" style="display:none;overflow:auto;padding:12px">
                <label>Result table <select id="national-select"></select></label>
                <p id="national-count"></p><div class="ind-table-wrap"><table class="ind-table" id="national-table"></table></div>
                <button id="national-prev">Previous</button> <button id="national-next">Next</button>
            </section>
            {PF_HTML}
        </div>
    </div>
    
    <div id="about-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="document.getElementById('about-modal').style.display='none'">&times;</span>
            <h2>About the Data Sources</h2>
            <div id="about-content">{sources}</div>
        </div>
    </div>

    <script>
        // Data injected from Python
        const nationalTables = {json.dumps(tables, separators=(",", ":"), allow_nan=False)};
        const geojsonData = {json.dumps(geojson_data)};
        const tsData = {json.dumps(ts_data)};
        const portfolioData = {json.dumps(portfolios, separators=(',', ':'))};
        const extraMetrics = {json.dumps(extra_metrics)};
        const indiaData = {json.dumps(india_data, separators=(',', ':'))};
        
        const metricNames = {{
            'hill1': 'Crop Diversity (Effective N)',
            'resp_div': 'Response Diversity',
            'stability': 'Stability (1/CV)',
            'resistance': 'Resistance (Median ratio)',
            'recovery': 'Recovery (Median ratio)',
            'vulnerability': 'Vulnerability (% years < 0.9)',
            'phi_sync': 'Synchrony'
        }};
        

        
        const metricDescriptions = {{
            'hill1': 'The effective number of crops grown (exponent of Shannon entropy). Higher values indicate a more diverse crop portfolio.',
            'resp_div': 'Measures how differently a country\\'s or district\\'s crops react to drought. This is the value-weighted spread of the estimated crop slopes.',
            'stability': 'Inverse coefficient of variation (1/CV) of the aggregate agricultural value. Higher means less year-to-year fluctuation.',
            'resistance': 'The ability to withstand a drought shock (ratio of production value during a drought year compared to pre-drought years).',
            'recovery': 'The ability to bounce back post-drought (ratio of post-drought production value compared to the drought year).',
            'vulnerability': 'The percentage of years where aggregate agricultural value drops more than 10% below its long-term trend.',
            'phi_sync': 'Crop synchrony. Measures how correlated crop value fluctuations are. Lower synchrony generally improves stability.'
        }};
        
        Object.keys(extraMetrics).forEach(k => {{
            metricNames[k] = extraMetrics[k][0];
            metricDescriptions[k] = extraMetrics[k][1];
        }});

        let currentScope = 'global';
        let currentMetric = 'hill1';
        let currentCountry = null;
        let currentDistrict = null;
        let currentIndiaState = 'ALL';
        
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
        if (indiaData && indiaData.extra_metrics) {{
            Object.keys(indiaData.extra_metrics).forEach(k => {{
                metricNames[k] = indiaData.extra_metrics[k][0];
                metricDescriptions[k] = indiaData.extra_metrics[k][1];
                colorscales[k] = indiaData.extra_metrics[k][2];
            }});
            document.getElementById('india-jump-btn').style.display = 'inline-block';
        }} else {{
            document.getElementById('scope-select').style.display = 'none';
            document.getElementById('scope-label').style.display = 'none';
        }}

        function getIndiaWindow() {{
            return currentScope === 'india_ext5' ? 'ext5' : 'core';
        }}

        function syncMetricOptions() {{
            const isIndia = currentScope !== 'global';
            const sel = document.getElementById('metric-select');
            Array.from(sel.options).forEach(opt => {{
                const sc = opt.getAttribute('data-scope');
                if (sc === 'global') opt.style.display = isIndia ? 'none' : '';
                else if (sc === 'india') opt.style.display = isIndia ? '' : 'none';
            }});
            const curOpt = sel.querySelector(`option[value="${{currentMetric}}"]`);
            if (curOpt && curOpt.style.display === 'none') {{
                currentMetric = 'hill1';
                sel.value = 'hill1';
            }}
        }}

        function populateIndiaFilters() {{
            if (!indiaData) return;
            const win = getIndiaWindow();
            const mDict = indiaData.metrics[win] || {{}};
            const stateSel = document.getElementById('india-state-select');
            const distSel = document.getElementById('india-dist-select');
            const states = Array.from(new Set(Object.keys(mDict).map(c => (indiaData.meta[c] || {{}}).state).filter(Boolean))).sort();
            stateSel.innerHTML = '<option value="ALL">All States (' + Object.keys(mDict).length + ' districts)</option>' +
                states.map(s => `<option value="${{s}}">${{s}}</option>`).join('');
            if (!states.includes(currentIndiaState)) currentIndiaState = 'ALL';
            stateSel.value = currentIndiaState;
            populateIndiaDistrictDropdown();
        }}

        function populateIndiaDistrictDropdown() {{
            if (!indiaData) return;
            const win = getIndiaWindow();
            const mDict = indiaData.metrics[win] || {{}};
            const distSel = document.getElementById('india-dist-select');
            const codes = Object.keys(mDict).filter(c => {{
                const st = (indiaData.meta[c] || {{}}).state;
                return currentIndiaState === 'ALL' || st === currentIndiaState;
            }}).sort((a, b) => {{
                const ma = indiaData.meta[a] || {{}}, mb = indiaData.meta[b] || {{}};
                return (ma.state + ' - ' + ma.name).localeCompare(mb.state + ' - ' + mb.name);
            }});
            distSel.innerHTML = `<option value="">Select a district (${{codes.length}})...</option>` +
                codes.map(c => {{
                    const m = indiaData.meta[c] || {{}};
                    return `<option value="${{c}}">${{m.state}} &mdash; ${{m.name}} (#${{c}})</option>`;
                }}).join('');
            if (!codes.includes(currentDistrict)) currentDistrict = null;
            distSel.value = currentDistrict || '';
        }}

        function switchScope(scope) {{
            if (scope !== 'global' && !indiaData) return;
            currentScope = scope;
            document.getElementById('scope-select').value = scope;
            const isIndia = scope !== 'global';
            document.getElementById('india-filters').style.display = isIndia ? 'flex' : 'none';
            document.getElementById('india-jump-btn').style.display = (!isIndia && indiaData) ? 'inline-block' : 'none';
            syncMetricOptions();
            if (isIndia) {{
                populateIndiaFilters();
                if (!currentDistrict || !indiaData.metrics[getIndiaWindow()][currentDistrict]) {{
                    document.getElementById('instruction').style.display = 'block';
                    document.getElementById('instruction').innerText = 'Click an Indian district on the map (or select one above) to view district timeseries';
                    document.getElementById('district-summary').style.display = 'none';
                    document.getElementById('chart-explanation').style.display = 'none';
                    document.getElementById('line-chart').style.visibility = 'hidden';
                    document.getElementById('bar-chart').style.visibility = 'hidden';
                }} else {{
                    renderDistrictCharts(currentDistrict);
                }}
            }} else {{
                document.getElementById('district-summary').style.display = 'none';
                if (currentCountry) {{
                    renderCharts(currentCountry);
                }} else {{
                    document.getElementById('instruction').style.display = 'block';
                    document.getElementById('instruction').innerText = 'Click a country on the map to view timeseries';
                    document.getElementById('chart-explanation').style.display = 'none';
                    document.getElementById('line-chart').style.visibility = 'hidden';
                    document.getElementById('bar-chart').style.visibility = 'hidden';
                }}
            }}
            if (activeTab === 'pf') renderPortfolio();
            renderMap();
        }}

        const percent = v => v == null ? 'n/a' : (100 * v).toFixed(1) + '%';
        function mapValue(value) {{
            if (value == null) return null;
            return ['vulnerability','vuln_wet','vuln_drought'].includes(currentMetric) ? 100 * value : value;
        }}

        function renderMap() {{
            document.getElementById('metric-description').innerText = metricDescriptions[currentMetric] || '';
            if (currentScope !== 'global' && indiaData) {{
                renderIndiaMap();
                return;
            }}
            
            // Extract z values
            const zValues = [];
            const isoCodes = [];
            const hoverTexts = [];
            
            geojsonData.features.forEach(f => {{
                if (f.properties && f.properties[currentMetric] !== undefined) {{
                    zValues.push(mapValue(f.properties[currentMetric]));
                    isoCodes.push(f.properties.iso3);
                    const value = f.properties[currentMetric];
                    hoverTexts.push('<b>' + f.properties.iso3 + '</b><br>' + metricNames[currentMetric] +
                        ': ' + fmtNum(mapValue(value)));
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

        function renderIndiaMap() {{
            const win = getIndiaWindow();
            const mDict = indiaData.metrics[win] || {{}};
            const zValues = [], distCodes = [], hoverTexts = [];
            const bgCodes = [], bgTexts = [];
            const fmt = (v, d=2) => (v === null || v === undefined || Number.isNaN(v)) ? 'n/a' : Number(v).toFixed(d);

            indiaData.geojson.features.forEach(f => {{
                const code = f.properties.dist_code;
                const meta = indiaData.meta[code] || {{ name: code, state: 'Unknown' }};
                if (currentIndiaState !== 'ALL' && meta.state !== currentIndiaState) return;
                const m = mDict[code];
                if (m && m[currentMetric] !== null && m[currentMetric] !== undefined) {{
                    distCodes.push(code);
                    zValues.push(mapValue(m[currentMetric]));
                    hoverTexts.push(
                        `<b>${{meta.name}}</b> (${{meta.state}}) [#${{code}}]<br>` +
                        `${{metricNames[currentMetric]}}: <b>${{fmt(mapValue(m[currentMetric]), 3)}}</b><br>` +
                        `Hill-1: ${{fmt(m.hill1, 2)}} | Resp Div: ${{fmt(m.resp_div, 3)}}<br>` +
                        `Stability: ${{fmt(m.stability, 2)}} | Vuln: ${{percent(m.vulnerability)}} (Drought: ${{percent(m.vuln_drought)}})<br>` +
                        `Resistance: ${{fmt(m.resistance, 2)}} | Recovery: ${{fmt(m.recovery, 2)}}<br>` +
                        `Events: ${{m.n_events}} | Balanced Crops: ${{m.n_crops_value}} (${{percent(m.value_coverage)}})`
                    );
                }} else {{
                    bgCodes.push(code);
                    const status = win === 'core' ? meta.core_status : meta.ext5_status;
                    bgTexts.push(
                        `<b>${{meta.name}}</b> (${{meta.state}}) [#${{code}}]<br>` +
                        `Status: ${{status || 'no metric value'}}`
                    );
                }}
            }});

            const traces = [];
            if (bgCodes.length > 0) {{
                traces.push({{
                    type: 'choropleth',
                    geojson: indiaData.geojson,
                    featureidkey: 'properties.dist_code',
                    locations: bgCodes,
                    z: bgCodes.map(() => 0),
                    text: bgTexts,
                    hoverinfo: 'text',
                    colorscale: [[0, '#e5e8e8'], [1, '#e5e8e8']],
                    showscale: false,
                    marker: {{ line: {{ color: '#bdc3c7', width: 0.4 }} }}
                }});
            }}
            traces.push({{
                type: 'choropleth',
                geojson: indiaData.geojson,
                featureidkey: 'properties.dist_code',
                locations: distCodes,
                z: zValues,
                text: hoverTexts,
                hoverinfo: 'text',
                colorscale: colorscales[currentMetric] || 'Viridis',
                marker: {{ line: {{ color: '#fff', width: 0.5 }} }},
                colorbar: {{
                    title: metricNames[currentMetric],
                    thickness: 10,
                    len: 0.8,
                    x: 0.98,
                    y: 0.5
                }}
            }});

            const layout = {{
                geo: {{
                    showframe: false,
                    showcoastlines: false,
                    showland: false,
                    projection: {{ type: 'mercator' }},
                    fitbounds: 'locations'
                }},
                margin: {{ t: 0, b: 0, l: 0, r: 0 }},
                clickmode: 'event+select'
            }};

            Plotly.newPlot('map', traces, layout, {{responsive: true}});
            document.getElementById('map').on('plotly_click', function(data) {{
                if (data.points.length > 0) {{
                    const code = String(data.points[0].location);
                    if (mDict[code]) {{
                        document.getElementById('india-dist-select').value = code;
                        renderDistrictCharts(code);
                    }}
                }}
            }});
        }}
        
        function renderCharts(iso3) {{
            if (!tsData[iso3]) return;
            
            currentCountry = iso3;
            document.getElementById('instruction').style.display = 'none';
            document.getElementById('district-summary').style.display = 'none';
            const expEl = document.getElementById('chart-explanation');
            expEl.style.display = 'block';
            let indiaBanner = '';
            if (iso3 === 'IND' && indiaData) {{
                indiaBanner = `<div style="margin-top:6px;padding-top:6px;border-top:1px solid #bce0fd;">` +
                    `<strong>Phase 4 Subnational Analysis Available:</strong> India has ${{Object.keys(indiaData.metrics.core).length}} core districts (1998&ndash;2022) and ${{Object.keys(indiaData.metrics.ext5).length}} extended 5-crop districts (1993&ndash;2015). ` +
                    `<button onclick="switchScope('india_core')" style="background:#2c3e50;color:#fff;border:none;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:12px;">Open India District Map &rarr;</button></div>`;
            }}
            expEl.innerHTML =
                `<strong>Top Chart:</strong> Shows the yield anomalies (deviations from long-term trends) for the country's top 6 crops by value.<br>` +
                `<strong>Bottom Chart:</strong> Shows the Cropland-weighted SPEI-12 climate index. Negative values indicate drier conditions.<br>` +
                `<strong>Red Shaded Regions:</strong> Mark drought events in the calculated catalogue (SPEI-12 &le; -1.0) based on the project's criteria.<br>` +
                `<strong>Blue Shaded Regions:</strong> Represent wet events (SPEI-12 &ge; +1.0), the mirror of the drought rule. Bars are red when SPEI-12 &le; -1.0 and dark blue when SPEI-12 &ge; +1.0.` +
                indiaBanner;
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

        function renderDistrictCharts(distCode) {{
            if (!indiaData) return;
            const win = getIndiaWindow();
            const dts = (indiaData.ts[win] || {{}})[distCode];
            const m = (indiaData.metrics[win] || {{}})[distCode];
            const meta = (indiaData.meta || {{}})[distCode] || {{ name: distCode, state: '' }};
            if (!dts || !m) return;

            currentDistrict = distCode;
            document.getElementById('instruction').style.display = 'none';
            const sumEl = document.getElementById('district-summary');
            sumEl.style.display = 'block';
            const fmt = (v, d=2) => (v === null || v === undefined || Number.isNaN(v)) ? 'n/a' : Number(v).toFixed(d);
            const winLabel = win === 'core' ? 'Core DES Window (1998&ndash;2022, 47 crops)' : 'Extended 5-Crop Mirror (1993&ndash;2015)';
            const cropPills = Object.keys(dts.crop_meta || {{}}).map(c => {{
                const cm = dts.crop_meta[c];
                return `<span class="ds-pill"><strong>${{c}}</strong>: ${{percent(cm.share)}} val, &beta;<sub>SPEI</sub>=${{fmt(cm.beta, 3)}}</span>`;
            }}).join('');
            sumEl.innerHTML =
                `<div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;">` +
                `<strong style="font-size:15px;color:#2c3e50;">${{meta.name}}, ${{meta.state}} <span style="color:#7f8c8d;font-weight:normal;">(District #${{distCode}})</span></strong>` +
                `<span style="font-size:12px;color:#555;">${{winLabel}}</span></div>` +
                `<div class="ds-grid">` +
                `<div class="ds-item"><span class="lbl">Crop Diversity (Hill-1)</span><span class="val">${{fmt(m.hill1, 2)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Response Diversity</span><span class="val">${{fmt(m.resp_div, 3)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Stability (1/CV)</span><span class="val">${{fmt(m.stability, 2)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Resistance / Recovery</span><span class="val">${{fmt(m.resistance, 2)}} / ${{fmt(m.recovery, 2)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Vulnerability (All / Drought)</span><span class="val">${{percent(m.vulnerability)}} / ${{percent(m.vuln_drought)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Synchrony (&phi;<sub>sync</sub>)</span><span class="val">${{fmt(m.phi_sync, 3)}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Events / Drought Yrs</span><span class="val">${{m.n_events}} / ${{m.n_drought_years}}</span></div>` +
                `<div class="ds-item"><span class="lbl">Balanced Crops (Cov.)</span><span class="val">${{m.n_crops_value}} (${{percent(m.value_coverage)}})</span></div>` +
                `<div class="ds-item"><span class="lbl">Mean Irrigation Share</span><span class="val">${{m.irrig_share !== null ? fmt(m.irrig_share * 100, 1) + '%' : 'n/a'}}</span></div>` +
                `</div>` +
                (cropPills ? `<div class="ds-crops"><strong>Top balanced crops (value share &amp; drought slope &beta;):</strong> ${{cropPills}}</div>` : '');

            const expEl = document.getElementById('chart-explanation');
            expEl.style.display = 'block';
            expEl.innerHTML =
                `<strong>Top Chart:</strong> District aggregate production anomaly (<code>anom</code>, bold dashed black line) and top balanced crop yield anomalies.<br>` +
                `<strong>Bottom Chart:</strong> District cropland-weighted December <code>SPEI-12</code> bars, September Kharif <code>SPEI-4</code> (green dotted), and 5-year lagged time-varying diversity <code>hill1_tv</code> (right axis, purple line).<br>` +
                `<strong>Red Shaded Regions:</strong> District drought events in the calculated catalogue (<code>SPEI-12 &le; -1.0</code>, first year of a run with 3 drought-free pre-years).`;

            document.getElementById('line-chart').style.visibility = 'visible';
            document.getElementById('bar-chart').style.visibility = 'visible';

            const years = dts.years;
            const shapes = [];
            (dts.drought_years || []).forEach(year => {{
                shapes.push({{
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: year - 0.5, x1: year + 0.5,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(231, 76, 60, 0.16)',
                    line: {{ width: 0 }}
                }});
            }});

            // 1. Line Chart (District total anomaly + top crop yield anomalies)
            const lineData = [{{
                x: years,
                y: dts.anom,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'District Value Anomaly (anom)',
                line: {{ width: 2.5, color: '#111', dash: 'dash' }},
                marker: {{ size: 5, color: '#111' }}
            }}];
            Object.keys(dts.crops || {{}}).forEach(crop => {{
                lineData.push({{
                    x: years,
                    y: dts.crops[crop],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: crop,
                    line: {{ width: 1.8 }},
                    marker: {{ size: 4 }}
                }});
            }});

            const lineLayout = {{
                title: `${{meta.name}} (${{meta.state}}) &mdash; District Value &amp; Crop Yield Anomalies`,
                margin: {{ t: 40, b: 20, l: 55, r: 20 }},
                xaxis: {{ title: 'Year', showgrid: true }},
                yaxis: {{ title: 'Relative Anomaly', zeroline: true, zerolinecolor: '#999' }},
                shapes: shapes,
                legend: {{ orientation: 'h', y: -0.22 }}
            }};
            Plotly.newPlot('line-chart', lineData, lineLayout, {{responsive: true}});

            // 2. Bar Chart (District SPEI-12 + Kharif SPEI-4 + time-varying hill1_tv)
            const barColors = dts.spei.map(v => v !== null && v <= -1.0 ? '#e74c3c' : (v !== null && v >= 1.0 ? '#1f4e9c' : '#aab7b8'));
            const barData = [{{
                x: years,
                y: dts.spei,
                type: 'bar',
                marker: {{ color: barColors }},
                name: 'Dec SPEI-12 (spei12_w)'
            }}];
            if ((dts.spei_kh || []).some(v => v !== null)) {{
                barData.push({{
                    x: years,
                    y: dts.spei_kh,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Kharif SPEI-4 (Sept)',
                    line: {{ color: '#16a085', width: 1.8, dash: 'dot' }},
                    marker: {{ size: 4, color: '#16a085' }}
                }});
            }}
            if ((dts.hill1_tv || []).some(v => v !== null)) {{
                barData.push({{
                    x: years,
                    y: dts.hill1_tv,
                    yaxis: 'y2',
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Lagged Diversity (hill1_tv, t-5..t-1)',
                    line: {{ color: '#8e44ad', width: 2.2 }},
                    marker: {{ size: 5, color: '#8e44ad', symbol: 'diamond' }}
                }});
            }}

            const barLayout = {{
                title: `${{meta.name}} &mdash; District Drought Exposure (SPEI) &amp; Lagged Diversity (hill1_tv)`,
                margin: {{ t: 40, b: 45, l: 55, r: 55 }},
                xaxis: {{ title: 'Year' }},
                yaxis: {{ title: 'SPEI', zeroline: true, zerolinecolor: '#999' }},
                yaxis2: {{ title: 'Lagged Hill-1 (hill1_tv)', overlaying: 'y', side: 'right', showgrid: false }},
                shapes: shapes,
                legend: {{ orientation: 'h', y: -0.25 }}
            }};
            Plotly.newPlot('bar-chart', barData, barLayout, {{responsive: true}});
        }}
        
        document.getElementById('scope-select').addEventListener('change', function(e) {{
            switchScope(e.target.value);
        }});

        document.getElementById('metric-select').addEventListener('change', function(e) {{
            currentMetric = e.target.value;
            renderMap();
        }});

        document.getElementById('india-state-select').addEventListener('change', function(e) {{
            currentIndiaState = e.target.value;
            switchScope(currentScope);
        }});

        document.getElementById('india-dist-select').addEventListener('change', function(e) {{
            currentDistrict = e.target.value || null;
            switchScope(currentScope);
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
        
    content = html_template.encode("utf-8")
    size_mb = len(content) / (1024 * 1024)
    if size_mb >= MAX_SIZE_MB:
        raise ValueError(f"Explorer is {size_mb:.2f} MB, limit {MAX_SIZE_MB} MB")
    with NamedTemporaryFile(dir=out_file.parent, delete=False) as handle:
        staged = Path(handle.name)
    try:
        staged.write_bytes(content)
        staged.chmod(out_file.stat().st_mode & 0o777 if out_file.exists() else 0o644)
        staged.replace(out_file)
    finally:
        staged.unlink(missing_ok=True)
    print(f"Explorer built: {out_file} ({size_mb:.2f} MB)")
    print(f"Countries included: {len(included_iso3)}")
    print(f"Countries with a portfolio: {len(portfolios)}")
    if india_data:
        print(f"Phase 4 India districts: {len(india_data['metrics']['core'])} core, {len(india_data['metrics']['ext5'])} ext5")
    print(f"Extra map layers: {list(extra_metrics) or 'none'}")

if __name__ == "__main__":
    build_explorer()
