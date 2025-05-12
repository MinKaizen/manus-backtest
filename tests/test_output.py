# tests/test_output_consistency.py
import filecmp
import subprocess
import os

def test_output():
    expected_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    output_dir = os.path.join("./output")
    file_names = ["trade_analysis.txt", "trade_results.csv"]

    # Run the script (assumes it writes to files in the current working directory)
    subprocess.run(["python", "src/manus_backtest/main.py"], check=True)

    # Compare outputs
    for filename in file_names:
        expected = os.path.join(expected_dir, filename)
        actual = os.path.join(output_dir, filename)
        assert filecmp.cmp(expected, actual, shallow=False), f"{filename} differs from expected"

