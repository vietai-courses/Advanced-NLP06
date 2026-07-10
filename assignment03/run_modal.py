import modal

app = modal.App("evoagent-student")

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.13")
    .dockerfile_commands("ENTRYPOINT []")
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "pip3 install 'datasets>=2.18.0' 'pydantic>=2.5.0' "
        "'sentence-transformers>=2.7.0' 'matplotlib>=3.8.0' 'scipy>=1.12.0' "
        "'huggingface-hub>=0.22.0'",
        "python3 -c \"import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/torchao/quantization/quant_api.py'); content = p.read_text(encoding='utf-8'); content = content.replace('\\\\.', '\\\\\\\\.'); p.write_text(content, encoding='utf-8')\"",
    )
    .add_local_dir(
        ".",
        remote_path="/evoagent",
        ignore=[".venv", "runs", ".git", "__pycache__", ".pytest_cache", "*.pdf", "graders"]
    )
)

volume = modal.Volume.from_name("evoagent-runs", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=21600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run():
    """Self-optimization: Qwen guides itself."""
    import os
    import subprocess
    import shutil
    shutil.rmtree("/runs/exp_self", ignore_errors=True)
    os.chdir("/evoagent")
    subprocess.run(
        [
            "python", "main.py",
            "--T", "7",
            "--dataset", "local_financial_qa",
            "--output-dir", "/runs/exp_self",
            "--train-size", "300",
            "--dev-size", "584",
            "--model", "QuantTrio/Qwen3.5-4B-AWQ",
            "--gpu-memory-utilization", "0.7",
            "--progressive-reflections",
            "--afo-mode", "best",
        ],
        check=True,
    )
    
    # Generate evolution_proof.json
    import sys
    sys.path.insert(0, "/evoagent")
    from src.strategy import StrategyHistory
    from pathlib import Path
    import json
    
    history_path = Path("/runs/exp_self/history.jsonl")
    if history_path.exists():
        history = StrategyHistory(history_path)
        history.load()
        best_strategy = history.best_strategy()
        best_acc = best_strategy.metadata.dev_accuracy if best_strategy else 0.0
        
        proof_data = {
            "status": "success",
            "best_iteration": best_strategy.metadata.iteration if best_strategy else None,
            "best_dev_accuracy": best_acc,
            "baseline_accuracy": 0.42,
            "history": [
                {
                    "iteration": s.metadata.iteration,
                    "dev_accuracy": s.metadata.dev_accuracy
                }
                for s in history.strategies
            ]
        }
        proof_path = Path("/runs/exp_self/evolution_proof.json")
        with open(proof_path, "w", encoding="utf-8") as f:
            json.dump(proof_data, f, ensure_ascii=False, indent=2)
        print(f"Generated evolution proof in volume: {proof_path}")

        # Save best strategy as iter_best_strategy.json for convenient submission
        if best_strategy:
            best_strat_path = Path("/runs/exp_self/iter_best_strategy.json")
            best_strat_path.write_text(best_strategy.to_json(), encoding="utf-8")
            print(f"Saved best strategy (iter {best_strategy.metadata.iteration}, acc={best_acc:.3f}) to: {best_strat_path}")
    else:
        print("Error: history.jsonl not found. Cannot generate evolution proof.")
        
    volume.commit()

@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run_test():
    """Fast test run of the self-evolution loop to verify correctness."""
    import os
    import subprocess
    import shutil
    shutil.rmtree("/runs/exp_test", ignore_errors=True)
    os.chdir("/evoagent")
    subprocess.run(
        [
            "python", "main.py",
            "--T", "5",
            "--dataset", "local_financial_qa",
            "--output-dir", "/runs/exp_test",
            "--train-size", "32",
            "--dev-size", "32",
            "--model", "QuantTrio/Qwen3.5-4B-AWQ",
            "--gpu-memory-utilization", "0.7",
            "--skip-analysis",
        ],
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=32768,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run_sandbox():
    """Run Stage 0: Sandbox zero-shot baseline."""
    import sys
    sys.path.insert(0, "/evoagent")
    from src.sandbox import run_sandbox_accuracy_check, get_model
    from pathlib import Path
    import json
    
    # 1. Evaluate baseline accuracy on 50 dev examples
    model = get_model("QuantTrio/Qwen3.5-4B-AWQ")
    eval_res = run_sandbox_accuracy_check(model, dev_size=50)
    
    result = {
        "status": "success",
        "baseline_accuracy": eval_res.get("accuracy", 0.0),
        "num_correct": eval_res.get("num_correct", 0),
        "num_examples": eval_res.get("num_examples", 0),
        "message": f"Sandbox ran successfully on Modal. Baseline Accuracy: {eval_res.get('accuracy', 0.0)*100:.2f}% ({eval_res.get('num_correct')}/{eval_res.get('num_examples')})",
        "samples": eval_res.get("samples", []),
    }
    
    # Save to persistent Modal volume
    try:
        proof_path = Path("/runs/sandbox_proof.json")
        with open(proof_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Generated sandbox proof in volume: {proof_path}")
    except Exception as e:
        print(f"Error saving sandbox proof to volume: {e}")
        
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run_smoke():
    """Run the Stage 4 pre-flight smoke test."""
    import os
    import subprocess
    from pathlib import Path
    import json

    os.chdir("/evoagent")
    proc = subprocess.run(
        [
            "python", "main.py",
            "--smoke-test",
            "--dataset", "local_financial_qa",
            "--model", "QuantTrio/Qwen3.5-4B-AWQ",
            "--gpu-memory-utilization", "0.7",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    passed = proc.returncode == 0
    result = {
        "status": "success" if passed else "failed",
        "smoke_test_passed": passed,
        "returncode": proc.returncode,
        "message": "Modal smoke test passed." if passed else "Modal smoke test failed.",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    proof_path = Path("/runs/smoke_proof.json")
    proof_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated smoke proof in volume: {proof_path}")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    run.spawn()


@app.local_entrypoint()
def test():
    run_test.spawn()


@app.local_entrypoint()
def sandbox():
    print("Running sandbox prediction on Modal...")
    result = run_sandbox.remote()
    
    import json
    with open("sandbox_proof.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Successfully saved sandbox_proof.json locally!")


@app.local_entrypoint()
def get_proof():
    """Download the evolution proof and results from the Modal volume to your local directory."""
    import os
    import subprocess
    from pathlib import Path
    import shutil
    
    print("Downloading evolution results from Modal volume...")
    try:
        subprocess.run(["modal", "volume", "get", "evoagent-runs", "exp_self", "./runs"], check=True)
    except Exception as e:
        print(f"Error executing 'modal volume get': {e}")
        print("Please ensure modal CLI is authenticated and correct volume name is used.")
        return
        
    proof_src = Path("runs/exp_self/evolution_proof.json")
    if proof_src.exists():
        shutil.copy(proof_src, "evolution_proof.json")
        print("Successfully retrieved and saved local 'evolution_proof.json'!")
    else:
        print("Error: runs/exp_self/evolution_proof.json does not exist. Did you run 'modal run run_modal.py::main' successfully first?")


@app.local_entrypoint()
def smoke():
    print("Running Stage 4 smoke test on Modal...")
    result = run_smoke.remote()

    import json
    with open("smoke_proof.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Successfully saved smoke_proof.json locally!")

    if not result.get("smoke_test_passed"):
        raise RuntimeError(result.get("message", "Modal smoke test failed."))


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=7200,
    retries=1,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/runs": volume},
)
def run_submit(strategy_path: str, output_file: str = "/runs/submission.csv", limit: int = None):
    """Generate Kaggle test submission from a strategy."""
    import os
    import subprocess
    os.chdir("/evoagent")
    cmd = [
        "python", "submit.py",
        "--strategy-path", strategy_path,
        "--output-file", output_file,
        "--model", "QuantTrio/Qwen3.5-4B-AWQ",
        "--gpu-memory-utilization", "0.7",
        "--num-samples", "5",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(
        cmd,
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def submit(strategy_path: str, output_file: str = "submission.csv", limit: int = None):
    import os
    import subprocess
    import shutil
    from pathlib import Path

    if not output_file.startswith("/"):
        remote_output = f"/runs/{output_file}"
    else:
        remote_output = output_file

    print(f"Generating test set predictions on Modal via strategy: {strategy_path}...")
    run_submit.remote(strategy_path, remote_output, limit)

    # 1. Determine volume-relative path and local output path
    if remote_output.startswith("/runs/"):
        volume_rel_path = remote_output[len("/runs/"):]
    else:
        volume_rel_path = remote_output.lstrip("/")

    if output_file.startswith("/runs/"):
        local_output = Path(output_file).name
    else:
        local_output = output_file

    local_output_path = Path(local_output).resolve()
    local_dir = local_output_path.parent
    local_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path("temp_download")
    temp_dir.mkdir(exist_ok=True)

    # Download CSV file
    print(f"Downloading submission CSV: {volume_rel_path} -> {local_output_path}")
    try:
        subprocess.run(["modal", "volume", "get", "evoagent-runs", volume_rel_path, str(temp_dir)], check=True)
        downloaded_file = temp_dir / Path(volume_rel_path).name
        if downloaded_file.exists():
            if local_output_path.exists():
                os.remove(local_output_path)
            shutil.move(str(downloaded_file), str(local_output_path))
            print(f"Successfully downloaded and saved local submission to '{local_output}'!")
        else:
            print(f"Warning: Could not find downloaded file at {downloaded_file}")
    except Exception as e:
        print(f"Error downloading submission file from volume: {e}")

    # Download details JSON file
    remote_path_obj = Path(remote_output)
    remote_debug_path = remote_path_obj.parent / f"{remote_path_obj.stem}_details.json"
    
    if str(remote_debug_path).startswith("/runs/"):
        volume_debug_rel_path = str(remote_debug_path)[len("/runs/"):]
    else:
        volume_debug_rel_path = str(remote_debug_path).lstrip("/")

    local_debug_path = local_dir / f"{local_output_path.stem}_details.json"
    print(f"Downloading debug details JSON: {volume_debug_rel_path} -> {local_debug_path}")
    try:
        subprocess.run(["modal", "volume", "get", "evoagent-runs", volume_debug_rel_path, str(temp_dir)], check=True)
        downloaded_debug = temp_dir / Path(volume_debug_rel_path).name
        if downloaded_debug.exists():
            if local_debug_path.exists():
                os.remove(local_debug_path)
            shutil.move(str(downloaded_debug), str(local_debug_path))
            print(f"Successfully downloaded and saved local debug details to '{local_debug_path}'!")
        else:
            print(f"Warning: Could not find downloaded debug file at {downloaded_debug}")
    except Exception as e:
        print(f"Error downloading debug file from volume: {e}")

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

