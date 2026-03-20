from tests._bootstrap import ROOT

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from matcha import __version__
from matcha.example import _resolve_example_runtime_config
from matcha.resources import get_packaged_path
from matcha.run import _resolve_config_path, _resolve_lookup_table_paths, _resolve_resource_path


class ResourceTests(unittest.TestCase):
    def test_version_string_present(self) -> None:
        self.assertTrue(isinstance(__version__, str) and __version__)

    def test_packaged_data_files_exist(self) -> None:
        self.assertTrue(get_packaged_path("data/cs_l=3000_k=2500.mat").is_file())
        self.assertTrue(get_packaged_path("data/jl_zeros_l=3000_k=2500.mat").is_file())

    def test_config_alias_paths_resolve(self) -> None:
        self.assertTrue(Path(_resolve_config_path("config.yaml")).is_file())
        self.assertTrue(Path(_resolve_config_path("configs/config.yaml")).is_file())
        self.assertTrue(Path(_resolve_config_path("config_example.yaml")).is_file())
        self.assertTrue(Path(_resolve_config_path("configs/config_example.yaml")).is_file())

    def test_packaged_config_resolves_outside_repo(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                resolved = Path(_resolve_config_path("config.yaml"))
                self.assertTrue(resolved.is_file())
                self.assertIn("config", resolved.name)
            finally:
                os.chdir(old_cwd)

    def test_lookup_tables_resolve_from_packaged_config(self) -> None:
        config_path = _resolve_config_path("config.yaml")
        config = {
            "cs_path": "../data/cs_l=3000_k=2500.mat",
            "jl_zeros_path": "../data/jl_zeros_l=3000_k=2500.mat",
        }
        _resolve_lookup_table_paths(config=config, config_path=config_path)
        self.assertTrue(Path(config["cs_path"]).is_file())
        self.assertTrue(Path(config["jl_zeros_path"]).is_file())

    def test_example_template_resolves_from_working_directory_data(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "data").mkdir()
            example_map = tmp_path / "data" / "run2_class001.mrc"
            example_map.write_bytes(b"test")
            os.chdir(tmp_path)
            try:
                config_path = _resolve_config_path("config_example.yaml")
                resolved = _resolve_resource_path("data/run2_class001.mrc", config_path)
                self.assertEqual(Path(resolved), example_map)
            finally:
                os.chdir(old_cwd)

    def test_example_runtime_config_derives_template_metadata(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "data").mkdir()
            example_map = tmp_path / "data" / "run2_class001.mrc"
            example_map.write_bytes(b"test")
            os.chdir(tmp_path)
            try:
                config_path = _resolve_config_path("config_example.yaml")
                config = {
                    "path_template": "data/run2_class001.mrc",
                    "cs_path": "../data/cs_l=3000_k=2500.mat",
                    "jl_zeros_path": "../data/jl_zeros_l=3000_k=2500.mat",
                }
                with patch("matcha.run._read_template_meta", return_value=(96, 2.1)):
                    _resolve_example_runtime_config(config=config, config_path=config_path)
                self.assertEqual(config["N"], 96)
                self.assertEqual(config["voxel_size"], 2.1)
                self.assertEqual(config["random_seed"], 0)
                self.assertEqual(Path(config["path_template"]), example_map)
                self.assertTrue(Path(config["cs_path"]).is_file())
                self.assertTrue(Path(config["jl_zeros_path"]).is_file())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
