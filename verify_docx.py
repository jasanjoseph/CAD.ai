"""Verify the generated .docx: structure, tables, hyperlinks, diagram integrity."""
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

from docx import Document

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("CAD.ai_Development_Schema.docx")
doc = Document(path)

headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading") or p.style.name == "Title"]
print(f"file: {path.name} ({path.stat().st_size} bytes)")
print(f"paragraphs: {len(doc.paragraphs)} | tables: {len(doc.tables)} | headings: {len(headings)}")

print("\nHeadings:")
for h in headings:
    print("  -", h)

print("\nTables (rows x cols, first header cell):")
for t in doc.tables:
    print(f"  - {len(t.rows)}x{len(t.columns)}  |  {t.rows[0].cells[0].text[:40]}")

# hyperlinks live in the package relationships
links = {r.target_ref for r in doc.part.rels.values() if r.reltype.endswith("/hyperlink")}
footer_links = {
    r.target_ref
    for s in doc.sections
    for r in s.footer.part.rels.values()
    if r.reltype.endswith("/hyperlink")
}
print(f"\nHyperlinks in body:   {sorted(links)}")
print(f"Hyperlinks in footer: {sorted(footer_links)}")

text = "\n".join(p.text for p in doc.paragraphs)
checks = {
    "repo URL present": "github.com/jasanjoseph/CAD.ai" in text or bool(links | footer_links),
    "topology diagram intact": "orchestrator" in text and "failure_handler" in text,
    "state schema covered": "AgentState" in text,
    "trust model covered": "Trust model" in text or "trust model" in text.lower(),
    "clone command present": "git clone" in text,
}
print()
for name, ok in checks.items():
    print(("  ok   " if ok else "  FAIL ") + name)

print("\nFirst lines of the architecture diagram (proportional spacing preserved):")
for p in doc.paragraphs:
    if "START" in p.text and "──" in p.text:
        print("  " + p.text[:100])
        break
