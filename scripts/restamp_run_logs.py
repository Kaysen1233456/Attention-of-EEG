"""
运行日志「补写 / 重盖戳」工具 —— 更新元数据，但不破坏已有历史。

问题
----
`experiments_log/runs/` 里的日志有一部分是**事后回填**的（backfilled）。
回填时记录的 git 提交号，和后来实际改动后的提交号不一样。
如果直接覆盖 `meta.json`，原本记录的那个提交号就永久丢失了，
回溯时无法定位「当时跑的到底是哪一版代码」。

本脚本的做法
------------
把每一次记录到的提交号都**追加**进 `meta.json` 的 `sha_history` 数组，
只更新顶层的 `git` 字段，**永不删除 `sha_history` 里的既有条目**。

    首次运行：meta.git.sha  ->  迁移进 sha_history[0]
    之后运行：追加 sha_history[1], [2], ...

用法
----
    # 看所有日志当前的盖戳情况（不改任何文件）
    python scripts/restamp_run_logs.py --list

    # 预演：只显示会改什么，不落盘
    python scripts/restamp_run_logs.py --all --note "补上 B1 修复后的提交号" --dry-run

    # 实际执行
    python scripts/restamp_run_logs.py --all --note "补上 B1 修复后的提交号"

    # 只处理某一个
    python scripts/restamp_run_logs.py --run 20261009-150732_spectral_band_audit

退出码：0 成功；1 有错误；2 参数问题。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "experiments_log" / "runs"


def _git(*args):
    """在仓库根目录执行 git 命令；失败返回空字符串。"""
    try:
        out = subprocess.check_output(
            ("git",) + args,
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def git_info():
    sha = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": sha,
        "sha_short": sha[:7] if sha else "",
        "remote": _git("remote", "get-url", "origin"),
        "dirty": bool(porcelain),
        "status_porcelain": porcelain.splitlines() if porcelain else [],
    }


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def discover_runs():
    if not RUNS_DIR.is_dir():
        return []
    return sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())


def summarize(run_dir):
    """读取一个 run 的盖戳现状（只读，不落盘）。"""
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return {"run": run_dir.name, "error": "缺少 meta.json"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"run": run_dir.name, "error": "meta.json 解析失败: %s" % exc}

    history = meta.get("sha_history") or []
    return {
        "run": run_dir.name,
        "current_sha": (meta.get("git") or {}).get("sha_short", ""),
        "history_len": len(history),
        "history": [
            {"sha": (h or {}).get("sha_short", ""), "note": (h or {}).get("note", "")}
            for h in history
        ],
        "has_meta_git": bool((meta.get("git") or {}).get("sha")),
    }


def restamp_one(run_dir, note, dry_run=False):
    """重盖一个 run 的戳。返回 (changed: bool, message: str)。"""
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return False, "跳过（无 meta.json）"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "跳过（meta.json 解析失败: %s）" % exc

    current = git_info()
    stamp_time = now_iso()

    changed = False
    messages = []

    # --- sha_history 必须是数组，且只增不减 ---
    raw_history = meta.get("sha_history")
    if isinstance(raw_history, list):
        history = raw_history
    else:
        history = []
        if "sha_history" in meta:
            messages.append("sha_history 类型异常，已重建为数组")
        changed = True  # 首次引入该字段，属于变更

    known = {h.get("sha") for h in history if isinstance(h, dict) and h.get("sha")}

    # 1) 首次运行：把 meta.git 里原有的 sha 迁进历史，避免丢失
    old_git = meta.get("git") or {}
    old_sha = old_git.get("sha")
    if old_sha and old_sha not in known:
        history.append({
            "sha": old_sha,
            "sha_short": old_git.get("sha_short", old_sha[:7]),
            "stamped_at": meta.get("started_at") or meta.get("recorded_at") or stamp_time,
            "note": "initial (migrated from meta.git)",
        })
        known.add(old_sha)
        messages.append("迁移原 git.sha %s -> sha_history" % old_sha[:7])
        changed = True

    # 2) 追加当前 sha（若已存在则跳过，保证幂等）
    if current["sha"] and current["sha"] not in known:
        history.append({
            "sha": current["sha"],
            "sha_short": current["sha_short"],
            "stamped_at": stamp_time,
            "note": note or "restamp",
            "dirty": current["dirty"],
        })
        messages.append("追加当前 sha %s" % current["sha_short"])
        changed = True
    else:
        messages.append("当前 sha 已在历史中，不重复追加")

    meta["sha_history"] = history

    # 3) 顶层 git 更新为当前状态（这是「当前代码版本」，不是历史）
    if meta.get("git") != current:
        meta["git"] = current
        messages.append("更新顶层 git")
        changed = True

    if changed:
        meta["restamped_at"] = stamp_time

    if dry_run:
        return changed, "；".join(messages) + "（dry-run，未落盘）"

    if changed:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return True, "；".join(messages)

    return False, "；".join(messages)


def main():
    ap = argparse.ArgumentParser(
        description="补写运行日志元数据，保留 sha_history（只增不减）"
    )
    ap.add_argument("--list", action="store_true", help="列出所有 run 的盖戳现状（只读）")
    ap.add_argument("--all", action="store_true", help="处理 runs/ 下所有 run")
    ap.add_argument("--run", metavar="DIRNAME", help="只处理指定的 run 目录名")
    ap.add_argument("--note", default="", help="本次盖戳的原因（写进 sha_history）")
    ap.add_argument("--dry-run", action="store_true", help="只预演，不落盘")
    args = ap.parse_args()

    if not args.list and not args.all and not args.run:
        ap.print_help()
        return 2

    runs = discover_runs()
    if not runs:
        print("未找到任何 run 目录: %s" % RUNS_DIR)
        return 1

    # ---------- --list ----------
    if args.list:
        cur = git_info()
        print("仓库当前提交: %s (%s)%s" % (
            cur["sha_short"], cur["branch"], "  [有未提交改动]" if cur["dirty"] else ""
        ))
        print("run 目录: %s" % RUNS_DIR)
        print()
        print("%-46s %-10s %-8s %s" % ("RUN", "当前SHA", "历史条数", "历史"))
        print("-" * 110)
        for run_dir in runs:
            info = summarize(run_dir)
            if "error" in info:
                print("%-46s %s" % (info["run"], info["error"]))
                continue
            hist = " -> ".join(
                "%s(%s)" % (h["sha"], h["note"][:18]) for h in info["history"]
            ) or "(空)"
            print("%-46s %-10s %-8d %s" % (
                info["run"], info["current_sha"], info["history_len"], hist
            ))
        print()
        print("提示：`--all --note \"...\"` 可把当前提交号补进所有 run 的 sha_history。")
        return 0

    # ---------- 选择目标 ----------
    if args.run:
        targets = [r for r in runs if r.name == args.run]
        if not targets:
            print("找不到 run: %s" % args.run)
            print("可用: %s" % ", ".join(r.name for r in runs))
            return 1
    else:
        targets = runs

    # ---------- 执行 ----------
    n_changed = n_skipped = n_error = 0
    for run_dir in targets:
        try:
            changed, msg = restamp_one(run_dir, args.note, dry_run=args.dry_run)
        except Exception as exc:
            print("  [错误] %-46s %s" % (run_dir.name, exc))
            n_error += 1
            continue
        if changed:
            print("  [已更新] %-44s %s" % (run_dir.name, msg))
            n_changed += 1
        else:
            print("  [无变化] %-44s %s" % (run_dir.name, msg))
            n_skipped += 1

    print()
    print("完成：%d 更新 / %d 无变化 / %d 错误%s" % (
        n_changed, n_skipped, n_error, "（dry-run）" if args.dry_run else ""
    ))
    print("sha_history 为只增数组，既有历史条目未被改动。")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
