import json
import sys
import unittest
from pathlib import Path

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from graders.messages import print_stage_message


class TestModalSmokeProof(unittest.TestCase):
    def test_smoke_proof(self):
        print("\n--> Verifying smoke_proof.json...")
        proof_path = Path("smoke_proof.json")
        self.assertTrue(
            proof_path.exists(),
            "smoke_proof.json not found! Run 'modal run run_modal.py::smoke' first.",
        )

        try:
            with proof_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.fail(f"Failed to parse smoke_proof.json: {exc}")

        self.assertEqual(data.get("status"), "success", "Modal smoke run did not report success.")
        self.assertIs(data.get("smoke_test_passed"), True, "Smoke test did not pass.")
        self.assertEqual(data.get("returncode"), 0, "Smoke test process returned a non-zero exit code.")
        print("    [PASS] smoke_proof.json verified successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GRADER: MODAL SMOKE PROOF")
    print("=" * 60)

    suite = unittest.TestSuite()
    suite.addTest(TestModalSmokeProof("test_smoke_proof"))

    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 60)
        print("SUCCESS! MODAL SMOKE PROOF VERIFIED.")
        print("=" * 60 + "\n")
        print_stage_message("smoke")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("MODAL SMOKE PROOF GRADING FAILED.")
    print("=" * 60 + "\n")
    sys.exit(1)
