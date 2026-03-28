"""
Placeholder tests — real tests added as each module is built.
"""

def test_project_structure():
    """Verify required project files exist."""
    from pathlib import Path
    assert Path("generate_data.py").exists()
    assert Path("data_pipeline.py").exists()
    assert Path("analyst.py").exists()
    assert Path("app.py").exists()
    assert Path("requirements.txt").exists()


def test_requirements_has_key_packages():
    """Verify key packages are in requirements.txt."""
    content = Path("requirements.txt").read_text()
    for pkg in ["anthropic", "streamlit", "pydantic", "pandas", "numpy"]:
        assert pkg in content, f"{pkg} missing from requirements.txt"


from pathlib import Path