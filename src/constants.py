from pathlib import Path


standalone_dir = Path(__file__).parent.resolve()
root_dir = standalone_dir / ".."
images_dir = root_dir / "images"
data_dir = root_dir / "data"
props_dir = data_dir / "props"
dvrk_dir = data_dir / "dVRK"
psm_dir = dvrk_dir / "PSM"
