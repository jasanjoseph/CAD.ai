"""One-off: verify the GitHub repo contents and scan for leaked secrets."""
import base64
import json
import re
import subprocess
import sys

GH = r"C:\Program Files\GitHub CLI\gh.exe"
REPO = "jasanjoseph/CAD.ai"
PATTERNS = {
    "gemini key": re.compile(r"AQ\.[A-Za-z0-9_-]{20,}"),
    "openai key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "google api key": re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
}


def gh_api(path: str) -> dict | list:
    out = subprocess.run([GH, "api", path], capture_output=True)
    if out.returncode != 0:
        sys.exit(f"gh api {path} failed: {out.stderr.decode(errors='replace')[:200]}")
    return json.loads(out.stdout.decode("utf-8", errors="replace"))


tree = gh_api(f"repos/{REPO}/git/trees/main?recursive=1")
files = [t for t in tree["tree"] if t["type"] == "blob"]
print(f"REPO: {REPO}  branch: main  commit: {tree['sha'][:12]}")
print(f"{len(files)} files\n")
print(f"{'path':<42}{'size':>10}  sha")
total = 0
for t in files:
    total += t.get("size", 0)
    print(f"{t['path']:<42}{t.get('size', 0):>10}  {t['sha'][:8]}")
print(f"{'TOTAL':<42}{total:>10} bytes")

commits = gh_api(f"repos/{REPO}/commits")
print(f"\n{len(commits)} commit(s):")
for c in commits:
    print(f"  {c['sha'][:10]}  {c['commit']['message'].splitlines()[0]}")

print("\nSecret scan of every tracked file:")
leaks = 0
for t in files:
    content = ""
    try:
        blob = gh_api(f"repos/{REPO}/contents/{t['path']}")
        if blob.get("encoding") == "base64":
            content = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
    except SystemExit:
        content = ""
    hits = [name for name, pat in PATTERNS.items() if pat.search(content)]
    if hits:
        leaks += 1
        print(f"  !! {t['path']}: LEAKED {' & '.join(hits)}")
    else:
        print(f"  ok {t['path']}")
print(f"\n{'LEAKS FOUND: ' + str(leaks) if leaks else 'RESULT: no secrets in the repo'}")

readme = gh_api(f"repos/{REPO}/readme")
print(f"\nREADME renders: yes ({len(base64.b64decode(readme['content']))} bytes)")
meta = gh_api(f"repos/{REPO}")
print(f"visibility: {meta['visibility']} | description: {meta.get('description')}")
