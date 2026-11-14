"""Download the FAOSTAT Detailed Trade Matrix bulk file (about 420 MB) into data/raw/faostat/.

Source: FAOSTAT, Trade, Detailed trade matrix (bilateral trade flows by reporter and partner).
        https://bulks-faostat.fao.org/production/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip
`download()` has no resume: if the connection drops, run the script again.

Run: python -m src.crops.download_trade_matrix
"""
from ..common.sources import download
from .download import FAO_DIR, FAO_BASE

TM_FILE = "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"


def main():
    dest = download("FAOSTAT TM (Detailed trade matrix)", FAO_BASE + TM_FILE, FAO_DIR / TM_FILE,
                    note="Bilateral trade flows by reporter and partner (Normalized bulk file)")
    assert dest.stat().st_size > 1e6, TM_FILE
    print(f"\nFAOSTAT trade matrix: {dest.name}, {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
