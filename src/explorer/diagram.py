import zlib
import base64
import requests
from src.common.paths import FIGURES, USE_DUMMY

MERMAID_CODE = """
graph LR
    classDef source fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    F["FAOSTAT crop, land, food and trade data<br/>WDI national controls"]:::source --> C["National crop values, control variables,<br/>trade flows and calories"]:::data
    H["CRU temperature<br/>GGCMI crop calendars"]:::source --> G["National heat and seasonal exposure"]:::data
    S["SPEIbase and<br/>MapSPAM crop areas"]:::source --> G
    S --> GS["Country and crop-specific drought"]:::data
    C --> N["National metrics and models<br/>1993–2023"]:::process
    G --> N
    GS --> N
    S --> X["District climate<br/>and geographic matches"]:::data
    P["geoBoundaries districts"]:::source --> X
    A["DES area and production"]:::source --> V["District metrics and models<br/>1998–2022 core and 1993–2015 five-crop"]:::process
    B["Five-crop ICRISAT mirror<br/>and irrigation controls"]:::source --> V
    X --> V
    N --> R[Model tables and figures]:::output
    V --> R
    N --> E[Standalone explorer]:::output
    V --> E
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

