from tests._bootstrap import ROOT

import os
import subprocess
import sys
import unittest


class CliHelpTests(unittest.TestCase):
    def _run_help(self, module_name: str) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT / 'src')
        result = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_matcha_help(self) -> None:
        self._run_help("matcha")

    def test_matcha_example_help(self) -> None:
        self._run_help("matcha.example")


if __name__ == "__main__":
    unittest.main()
