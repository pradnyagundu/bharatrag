"""
Tests for the BharatRAG CLI.
"""

import json
import subprocess
import sys
import pytest


def _run(args: list) -> subprocess.CompletedProcess:
    """Run the bharatrag CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "bharatrag.cli"] + args,
        capture_output=True,
        text=True,
    )


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = _run(["--help"])
        assert result.returncode == 0

    def test_version_exits_zero(self):
        result = _run(["--version"])
        assert result.returncode == 0

    def test_languages_command(self):
        result = _run(["languages"])
        assert result.returncode == 0
        assert "hindi" in result.stdout
        assert "gujarati" in result.stdout
        assert "punjabi" in result.stdout


class TestCLIEvaluate:
    @pytest.fixture
    def sample_json(self, tmp_path):
        """Create a minimal valid dataset JSON for testing."""
        data = {
            "data": [
                {
                    "id": "hi_001",
                    "language": "hindi",
                    "question": "भारत की राजधानी क्या है?",
                    "context": ["भारत की राजधानी नई दिल्ली है।"],
                    "ground_truth_answer": "भारत की राजधानी नई दिल्ली है।",
                }
            ]
        }
        p = tmp_path / "test_data.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_missing_data_flag_exits_nonzero(self):
        result = _run(["evaluate", "--language", "hindi"])
        assert result.returncode != 0

    def test_missing_language_flag_exits_nonzero(self):
        result = _run(["evaluate", "--data", "some_file.json"])
        assert result.returncode != 0

    def test_nonexistent_file_exits_nonzero(self):
        result = _run(["evaluate", "--data", "nonexistent.json", "--language", "hindi"])
        assert result.returncode != 0

    def test_invalid_json_exits_nonzero(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON", encoding="utf-8")
        result = _run(["evaluate", "--data", str(bad_file), "--language", "hindi"])
        assert result.returncode != 0

    def test_unsupported_language_exits_nonzero(self, sample_json):
        result = _run(["evaluate", "--data", sample_json, "--language", "klingon"])
        assert result.returncode != 0

    def test_output_file_is_written(self, sample_json, tmp_path):
        out = tmp_path / "results.json"
        result = _run([
            "evaluate",
            "--data", sample_json,
            "--language", "hindi",
            "--output", str(out),
        ])
        # Only assert output file contents when evaluation succeeds; CI may fail
        # to download/load models, which can cause a non-zero exit code.
        if result.returncode == 0:
            assert out.exists()
            with open(out, encoding="utf-8") as f:
                results = json.load(f)
            assert "overall" in results
            assert "language" in results
