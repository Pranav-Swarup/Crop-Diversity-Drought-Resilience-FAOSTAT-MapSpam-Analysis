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

    subgraph Raw Sources
        F[FAOSTAT]:::source
        S[SPEIbase]:::source
        M[MapSPAM]:::source
        W[WDI]:::source
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
    
    subgraph Analysis
        T1 --> ME[Metrics Engine]:::process
        T2 --> ME
        T3 --> ME
    end
    
    subgraph Outputs
        ME --> R[Results & Figures]:::output
        ME --> E[Interactive Explorer]:::output
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
        response = requests.get(url)
        response.raise_for_status()
        
        out_path = deck_dir / f"pipeline.{fmt}"
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"Saved {out_path.name} to {deck_dir}")

if __name__ == "__main__":
    generate_diagram()
