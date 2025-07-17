import subprocess

def test_train_script_runs():
    """Test that train.py runs and prints accuracy."""
    result = subprocess.run(
        ["python", "src/train.py", "--test-size", "0.2", "--random-state", "42"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    # Check that accuracy is printed and is a float between 0 and 1
    lines = result.stdout.splitlines()
    acc_line = next((l for l in lines if "Accuracy:" in l), None)
    assert acc_line is not None
    acc = float(acc_line.split(":")[1])
    assert 0.0 <= acc <= 1.0