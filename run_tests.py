import subprocess
import sys

"""
Due to TensorFlow's stateful nature, running tests concurrently within a single process using the standard ``pytest test`` command
will result in errors (`ValueError: tf.function only supports singleton tf.Variables created on the first call`).
Each test file must be executed in an isolated process to ensure a clean state.
"""

test_files = [
    "test/test_train_adapter.py",
    "test/test_test_module.py",
    "test/test_generate_samples.py",
    "test/test_train_cli.py",
]

print("Running tests...")

for test_file in test_files:
    print("-" * 50)
    print(f"Test: {test_file}")

    result = subprocess.run([sys.executable, "-m", "pytest", test_file], check=False)

    if result.returncode != 0:
        print(f"Error: Tests in {test_file} failed.")
        sys.exit(result.returncode)

print("-" * 50)
print("All tests passed successfully!")