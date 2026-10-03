import shutil
from src.common.paths import FIGURES, USE_DUMMY

def collect_figures():
    print("Collecting deck figures...")
    
    deck_dir = FIGURES / "deck"
    if USE_DUMMY:
        deck_dir = FIGURES / "deck" / "dummy"
        
    deck_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old numbered pngs
    for f in deck_dir.glob("*.png"):
        if f.name.startswith("pipeline"):
            continue
        f.unlink()
    
    index_file = deck_dir / "INDEX.md"
    if index_file.exists():
        index_file.unlink()
        
    source_dirs = ["climate", "data", "results", "validation"]
    collected = []
    
    for d in source_dirs:
        d_path = FIGURES / d
        if USE_DUMMY:
            d_path = d_path / "dummy"
            
        if not d_path.exists():
            continue
            
        for img in sorted(d_path.glob("*.png")):
            collected.append({
                "owner": d,
                "orig_name": img.name,
                "orig_path": img
            })
            
    # Add pipeline
    pipeline_png = deck_dir / "pipeline.png"
    if pipeline_png.exists():
        collected.insert(0, {
            "owner": "explorer",
            "orig_name": "pipeline.png",
            "orig_path": pipeline_png
        })
        
    index_lines = ["# Presentation Deck Index\n", "| File | Owner | Description |", "|---|---|---|"]
    
    # 1-indexed for the figures, but pipeline can be 00
    counter = 1
    for item in collected:
        if item["orig_name"] == "pipeline.png":
            new_name = "00_pipeline.png"
        else:
            new_name = f"{counter:02d}_{item['orig_name']}"
            counter += 1
            
        dest = deck_dir / new_name
        
        if item["orig_path"] != dest:
            shutil.copy2(item["orig_path"], dest)
            
        desc = item["orig_name"].replace(".png", "").replace("_", " ").title()
        index_lines.append(f"| `{new_name}` | {item['owner']} | {desc} |")
        
    with open(index_file, "w") as f:
        f.write("\n".join(index_lines) + "\n")
        
    print(f"Collected {len(collected)} figures to {deck_dir}")

if __name__ == "__main__":
    collect_figures()
