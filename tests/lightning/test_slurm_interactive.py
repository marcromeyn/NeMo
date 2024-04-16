from unittest.mock import patch

import nemo.lightning  # monkey patch in here
import pytest
from lightning_fabric.plugins.environments import slurm


@pytest.mark.parametrize("job_name,expected", [
    ("bash", True),
    ("interactive", True),
    ("myjob_bash", True),
    ("session_interactive", True),
    ("myjob", False),
    ("session", False)
])
def test_is_slurm_interactive_mode(job_name, expected):    
    with patch('lightning_fabric.plugins.environments.slurm.SLURMEnvironment.job_name', return_value=job_name):
        assert nemo.lightning._is_slurm_interactive_mode() == expected   # noqa: SLF001
        assert slurm._is_slurm_interactive_mode() == expected   # noqa: SLF001
