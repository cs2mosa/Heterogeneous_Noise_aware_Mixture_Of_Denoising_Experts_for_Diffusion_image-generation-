import sys, inspect
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import models.model_config_v3 as mc3
print(inspect.getsource(mc3.preconditioned_HDMOEM_v3.__init__))
