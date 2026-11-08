"""Thread count for model fitting. Set MODEL_N_JOBS on small machines (e.g. 2 on the server)."""
import os

N_JOBS: int = int(os.environ.get("MODEL_N_JOBS", "4"))
