import zlib
import base64
import requests
from src.common.paths import FIGURES, USE_DUMMY

MERMAID_CODE = """
graph TD
    classDef source fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    subgraph Global Sources
        F[FAOSTAT]:::source
        S[SPEIbase]:::source
        M[MapSPAM]:::source
        W[WDI & CRU]:::source
    end

    subgraph Cleaning Pipelines
        F --> C1[crops.py]:::process
        W --> C1
        S --> C2[aggregate.py]:::process
        M --> C3[cropland.py]:::process
        C3 --> C2
    end

    subgraph Data Contracts
        C1 --> T1[(crops.parquet)]:::data
        C2 --> T2[(climate.parquet)]:::data
        C1 --> T3[(controls.parquet)]:::data
    end

    subgraph Phase 4 India Districts
        D1[DES District APY]:::source --> IC[india/clean.py & geo.py]:::process
        D2[ICRISAT Mirror]:::source --> IC
        D3[geoBoundaries ADM2]:::source --> IC
        S --> ICL[india/climate.py]:::process
        C3 --> ICL
        IC --> IP[india/panel.py]:::process
        ICL --> IP
    end

    subgraph Analysis
        T1 --> ME[Metrics & Panel Engine]:::process
        T2 --> ME
        T3 --> ME
    end

    subgraph Outputs
        ME --> R[Results & Figures]:::output
        IP --> R
        ME --> E[Interactive Explorer]:::output
        IP --> E
    end
"""

def generate_diagram():
    print("Generating pipeline diagram via Kroki API...")
    
    # Kroki encoding: zlib compress + base64 urlsafe
    compressed = zlib.compress(MERMAID_CODE.encode('utf-8'), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
    
    deck_dir = FIGURES / "deck"
    if USE_DUMMY:
        deck_dir = FIGURES / "deck" / "dummy"
    deck_dir.mkdir(parents=True, exist_ok=True)
    
    # We want SVG and PNG
    for fmt in ["svg", "png"]:
        url = f"https://kroki.io/mermaid/{fmt}/{encoded}"
        out_path = deck_dir / f"pipeline.{fmt}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)
            print(f"Saved {out_path.name} to {deck_dir}")
        except requests.RequestException as exc:
            if out_path.exists():
                print(f"Warning: Kroki request failed ({exc}); keeping existing {out_path.name}")
            else:
                raise

if __name__ == "__main__":
    generate_diagram()

