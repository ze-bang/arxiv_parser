import os

import pytest


def test_env_example_exists():
    assert os.path.exists(".env.example")
