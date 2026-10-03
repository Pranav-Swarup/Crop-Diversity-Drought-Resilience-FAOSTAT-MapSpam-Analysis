import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.common.paths import COUNTRIES_GEOJSON, METRICS, CROP_ANOMALIES, CLIMATE, EVENTS, SOURCES_MD, EXPLORER, USE_DUMMY

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
            'crops': crops_data
        }

    # 5. HTML Template
    print("  Generating HTML...")
    
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
        #charts-container {{ flex: 1; display: flex; flex-direction: column; padding: 10px; overflow-y: auto; background: #fafafa; }}
        .chart-wrapper {{ flex: 1; min-height: 300px; margin-bottom: 10px; background: white; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }}
        .modal-content {{ background: white; margin: 5% auto; padding: 30px; width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); line-height: 1.6; }}
        .close {{ float: right; cursor: pointer; font-size: 28px; font-weight: bold; color: #7f8c8d; margin-top: -10px; }}
        .close:hover {{ color: #2c3e50; }}
        
        .instruction-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255,255,255,0.9); padding: 20px; border-radius: 8px; border: 1px solid #ddd; text-align: center; pointer-events: none; z-index: 10; font-size: 18px; color: #555; }}
        #charts-inner {{ position: relative; width: 100%; height: 100%; display: flex; flex-direction: column; }}
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
                    </select>
                </div>
                <div id="metric-description"></div>
            </div>
            <div id="map"></div>
        </div>
        
        <div id="charts-container">
            <div id="charts-inner">
                <div id="instruction" class="instruction-overlay">Click a country on the map to view timeseries</div>
                <div id="chart-explanation">
                    <strong>Top Chart:</strong> Shows the yield anomalies (deviations from long-term trends) for the country's top 6 crops by value.<br>
                    <strong>Bottom Chart:</strong> Shows the Cropland-weighted SPEI-12 climate index. Negative values indicate drier conditions.<br>
                    <strong>Red Shaded Regions:</strong> Represent verified drought events (SPEI-12 &le; -1.0) based on the project's criteria.
                </div>
                <div id="line-chart" class="chart-wrapper" style="visibility: hidden;"></div>
                <div id="bar-chart" class="chart-wrapper" style="visibility: hidden;"></div>
            </div>
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
            const barColors = countryData.spei.map(v => v <= -1.0 ? '#e74c3c' : '#3498db');
            
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
        }}
        
        document.getElementById('metric-select').addEventListener('change', function(e) {{
            currentMetric = e.target.value;
            renderMap();
        }});
        
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
        
    with open(out_file, "w") as f:
        f.write(html_template)
        
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Explorer built: {out_file} ({size_mb:.2f} MB)")
    print(f"Countries included: {len(included_iso3)}")

if __name__ == "__main__":
    build_explorer()
