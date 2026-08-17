"""Publish the validated Stage 05 static release from a Colab CLI VM.

Expected VM inputs:
  /content/stage05_release.zip
  /tmp/github_token

The archive contains only release files.  This script performs no model fitting,
no simulation, and no dependency installation.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


REPO = "batestguy/soccer-eurovsSA"
ARCHIVE = Path("/content/stage05_release.zip")
TOKEN_FILE = Path("/tmp/github_token")
PAYLOAD = Path("/content/stage05_release_payload")
CLONE = Path("/content/soccerdl_stage05_release_repo")
OUT = Path("/content/soccerdl_out")

REQUIRED = [
    "README.md",
    "SESSION_HANDOFF.md",
    "render.yaml",
    "spaces/app.py",
    "spaces/README.md",
    "spaces/requirements.txt",
    "spaces/data/posterior.nc",
    "spaces/data/prior.nc",
    "spaces/data/strength_trends.csv",
    "spaces/data/elite_fit.csv",
    "spaces/data/dag_checks.csv",
    "stages/05_app/app.py",
    "stages/05_app/README_SPACES.md",
    "stages/05_app/requirements.txt",
    "stages/05_app/validate_release.py",
    "stages/05_app/release_stage05.py",
]


def run(args, *, cwd=None, env=None, capture=False):
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def validate_payload() -> None:
    for relative in REQUIRED:
        path = PAYLOAD / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty release file: {relative}")

    forbidden = [
        path for path in PAYLOAD.rglob("*")
        if path.is_file() and (
            path.name.lower() == "githubtoken.txt"
            or "gp_lens" in path.name.lower()
            or path.suffix.lower() in {".pem", ".key"}
        )
    ]
    if forbidden:
        raise RuntimeError(f"forbidden release files: {[str(p) for p in forbidden]}")

    if (PAYLOAD / "spaces" / "app.py").read_bytes() != (
        PAYLOAD / "stages" / "05_app" / "app.py"
    ).read_bytes():
        raise RuntimeError("app.py parity failed inside release archive")
    if (PAYLOAD / "spaces" / "README.md").read_bytes() != (
        PAYLOAD / "stages" / "05_app" / "README_SPACES.md"
    ).read_bytes():
        raise RuntimeError("README parity failed inside release archive")

    source = (PAYLOAD / "spaces" / "app.py").read_text(encoding="utf-8").lower()
    if "import pymc" in source or "pm.sample(" in source:
        raise RuntimeError("runtime fitting code found in Space app")
    if source.count("with gr.tab(") != 8:
        raise RuntimeError("Space app does not contain exactly eight tabs")


def git_env(token: str) -> dict[str, str]:
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(os.environ)
    env.update({
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_CONFIG_KEY_1": "user.name",
        "GIT_CONFIG_VALUE_1": "soccerdl-bot",
        "GIT_CONFIG_KEY_2": "user.email",
        "GIT_CONFIG_VALUE_2": "soccerdl-bot@users.noreply.github.com",
    })
    return env


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    if not TOKEN_FILE.is_file():
        raise FileNotFoundError(TOKEN_FILE)

    token = TOKEN_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    if not token:
        raise RuntimeError("empty GitHub token first line")

    for target in (PAYLOAD, CLONE):
        if target.exists():
            shutil.rmtree(target)
    OUT.mkdir(parents=True, exist_ok=True)
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    safe_extract(ARCHIVE, PAYLOAD)
    validate_payload()

    env = git_env(token)
    run(["git", "clone", "--depth", "1", f"https://github.com/{REPO}.git", str(CLONE)],
        env=env, capture=True)

    # Replace the Space directory exactly so obsolete GP-lens files cannot
    # survive the release.  Preserve earlier build scripts under stages/05_app.
    remote_spaces = CLONE / "spaces"
    if remote_spaces.exists():
        shutil.rmtree(remote_spaces)
    shutil.copytree(PAYLOAD / "spaces", remote_spaces)

    remote_stage = CLONE / "stages" / "05_app"
    remote_stage.mkdir(parents=True, exist_ok=True)
    for name in [
        "app.py", "README_SPACES.md", "requirements.txt",
        "validate_release.py", "release_stage05.py",
    ]:
        shutil.copy2(PAYLOAD / "stages" / "05_app" / name, remote_stage / name)
    for name in ["README.md", "SESSION_HANDOFF.md", "render.yaml"]:
        shutil.copy2(PAYLOAD / name, CLONE / name)

    run(
        ["git", "add", "-A", "README.md", "SESSION_HANDOFF.md", "render.yaml", "spaces", "stages/05_app"],
        cwd=CLONE,
        env=env,
    )
    status = run(["git", "status", "--short"], cwd=CLONE, env=env, capture=True).stdout.strip()
    if not status:
        raise RuntimeError("release produced no Git changes")
    run(
        ["git", "commit", "-m", "stage 05 release: sync static app bundle and handoff"],
        cwd=CLONE,
        env=env,
        capture=True,
    )
    run(["git", "push", "origin", "HEAD"], cwd=CLONE, env=env)
    commit = run(["git", "rev-parse", "HEAD"], cwd=CLONE, env=env, capture=True).stdout.strip()

    (OUT / "stage05_release_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (OUT / "stage05_release_app_sha256.txt").write_text(
        sha256(CLONE / "spaces" / "app.py") + "\n", encoding="utf-8"
    )
    print("release files changed:")
    print(status)
    print("commit:", commit)
    print("app sha256:", sha256(CLONE / "spaces" / "app.py"))
    print("runtime fitting: absent")
    print("model refit: not run")


if __name__ == "__main__":
    main()
