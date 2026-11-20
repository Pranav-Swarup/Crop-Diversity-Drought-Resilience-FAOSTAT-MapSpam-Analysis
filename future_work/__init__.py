"""Studies extending the presented crop diversity results. Author: Pmjarvis."""

from pathlib import Path
from site import addsitedir

# Support the isolated dependency install documented for this extension.
_dependencies = Path(__file__).resolve().parent / "local" / "python"
if _dependencies.is_dir():
    addsitedir(str(_dependencies))
