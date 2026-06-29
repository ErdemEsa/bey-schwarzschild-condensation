from pathlib import Path
import py_compile


def test_all_python_files_compile():
    root = Path(__file__).resolve().parents[1]
    files = [p for p in (root / "code").rglob("*.py") if "__pycache__" not in p.parts]
    assert files, "No Python files found."
    for path in files:
        py_compile.compile(str(path), doraise=True)


def test_expected_project_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for name in ["code", "results", "figures", "tests"]:
        assert (root / name).exists(), f"Missing {name}/"
