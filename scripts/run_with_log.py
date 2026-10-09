"""Run any training/evaluation command and capture a complete, committable run log.

为什么需要这个脚本
------------------
项目的 .gitignore 忽略了 `logs/` 和 `*.log`，所以直接用 `> run.log` 产生的日志
无法提交到 GitHub，回溯时找不到。本脚本把日志写到 **`experiments_log/runs/`**
（该目录不被忽略），并为每次运行生成结构化元数据，保证「任何改动和训练都可回溯」。

用法
----
    # 直接跑
    python scripts/run_with_log.py --name gen5_nested_cv -- \
        python scripts/cv_spectral_cnn.py --folds 8

    # 给日志加备注
    python scripts/run_with_log.py --name teacher_b_calibration --note "固定 0.5 对照" -- \
        python scripts/calibrate_teacher_b_threshold.py --output artifacts/teacher_b_calibration

产出目录结构
------------
    experiments_log/runs/<YYYYMMDD-HHMMSS>_<name>/
        meta.json     机器可读：命令、git 状态、环境、耗时、退出码
        stdout.log    完整标准输出
        stderr.log    完整标准错误
        combined.log  按到达顺序合并的时间线（人工回溯主要看这个）
        README.md     人读摘要 + 关键结果占位

退出码与子进程一致，方便在 CI / 批处理里串联。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = REPO_ROOT / "experiments_log" / "runs"
SECRET_PATTERNS = [
    re.compile(r"(ghp_[A-Za-z0-9]{20,})"),
    re.compile(r"(github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
]


def run_git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(r"\1<REDACTED>", text)
    return text


def package_versions() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "torch", "sklearn", "mne", "pandas"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command with full committable logging.",
        add_help=True,
    )
    parser.add_argument("--name", required=True, help="本次运行的短名，用于目录名")
    parser.add_argument("--note", default="", help="人工备注，写进 meta.json 与 README")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--quiet", action="store_true", help="不把子进程输出回显到终端")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- 之后的完整命令")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("必须在 -- 之后提供要执行的命令")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name).strip("_") or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_root) / f"{stamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 环境与版本快照 -------------------------------------------------
    meta = {
        "run_name": args.name,
        "note": args.note,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_line": " ".join(command),
        "cwd": str(REPO_ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": {
            "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "sha": run_git("rev-parse", "HEAD"),
            "sha_short": run_git("rev-parse", "--short", "HEAD"),
            "remote": run_git("remote", "get-url", "origin"),
            "dirty": bool(run_git("status", "--porcelain")),
            "status_porcelain": run_git("status", "--porcelain").splitlines(),
        },
    }

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    combined_path = run_dir / "combined.log"

    started = time.time()
    exit_code = None
    error = None

    print(f"[run_with_log] 运行目录: {run_dir}", flush=True)
    print(f"[run_with_log] 命令: {' '.join(command)}", flush=True)

    try:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        with open(stdout_path, "w", encoding="utf-8", errors="replace") as out_f, \
             open(stderr_path, "w", encoding="utf-8", errors="replace") as err_f, \
             open(combined_path, "w", encoding="utf-8", errors="replace") as both_f:

            def write_both(tag: str, line: str) -> None:
                both_f.write(f"[{tag}] {line}")
                both_f.flush()

            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            import threading

            def pump(stream, file_obj, tag: str) -> None:
                for line in iter(stream.readline, ""):
                    file_obj.write(line)
                    file_obj.flush()
                    write_both(tag, line)
                    if not args.quiet:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                stream.close()

            threads = [
                threading.Thread(target=pump, args=(process.stdout, out_f, "stdout")),
                threading.Thread(target=pump, args=(process.stderr, err_f, "stderr")),
            ]
            for thread in threads:
                thread.start()
            exit_code = process.wait()
            for thread in threads:
                thread.join()
    except FileNotFoundError as exc:
        error = f"命令不存在: {exc}"
        exit_code = 127
        combined_path.write_text(error + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        exit_code = 1
        combined_path.write_text(error + "\n", encoding="utf-8")

    duration = time.time() - started
    meta.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 3),
            "exit_code": exit_code,
            "success": exit_code == 0,
            "error": error,
            "artifacts": {
                "stdout": str(stdout_path.relative_to(REPO_ROOT)),
                "stderr": str(stderr_path.relative_to(REPO_ROOT)),
                "combined": str(combined_path.relative_to(REPO_ROOT)),
            },
        }
    )

    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    status = "成功" if exit_code == 0 else f"失败(exit={exit_code})"
    readme = f"""# 运行日志：{args.name}

- **状态**：{status}
- **开始**：{meta['started_at']}
- **耗时**：{duration:.1f} 秒
- **命令**：`{meta['command_line']}`
- **git 提交**：`{meta['git']['sha_short']}`（{meta['git']['branch']}，{'有未提交改动' if meta['git']['dirty'] else '工作区干净'}）
- **备注**：{args.note or '（无）'}

## 文件

| 文件 | 内容 |
|---|---|
| `meta.json` | 机器可读的完整元数据（命令、环境、依赖版本、git 状态、退出码） |
| `stdout.log` | 完整标准输出 |
| `stderr.log` | 完整标准错误 |
| `combined.log` | 按到达顺序合并的时间线，人工回溯看这个 |

## 关键结论

> 运行结束后请手工补写：本次跑出了什么、关键指标是多少、结论是什么、是否改变了后续计划。

```
（待补写）
```

## 复现方式

```bash
{meta['command_line']}
```
"""
    (run_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n[run_with_log] {status}，耗时 {duration:.1f}s")
    print(f"[run_with_log] 日志已写入 {run_dir}")
    return exit_code if exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
