"""Generate ORIGINAL_RAPTOR_FILE_INVENTORY.csv/.json before separation."""
import csv, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_DIRS = ["rag","dpo","controller","graph","graph_search","mb_sac","surrogate",
               "experiments","data","results","docs","tools","configs","scripts","outputs",
               "llm","topology_dpo","RGNN_RL","AnalogGym","AutoCkt","baselines","archive","external"]
LEGACY_ROOT_FILES = ["README.md",".gitignore","bsim4v5.out","inspect_results.py",
                     "tmp_make_paired_ucbtopk.py","tmp_pair_mini100_surcal.py","_tmp_dpo_debug.py"]
BULK = {"results","outputs","data","RGNN_RL","AnalogGym","AutoCkt"}  # no per-file hashing (cloud hydration)

def category(p: Path) -> str:
    s, ext = str(p).replace("\\","/").lower(), p.suffix.lower()
    if "__pycache__" in s: return "cache"
    if ext == ".py": return "test" if "test" in p.name else "source_code"
    if ext == ".csv":
        if "rag_memory" in s: return "rag_memory"
        if "dpo" in s or "preference" in s: return "dpo_pairs"
        if "/results/" in s: return "experiment_result"
        return "csv_dataset"
    if ext in (".json",".jsonl"):
        if "rag_memory" in s: return "rag_memory"
        if "dpo" in s or "preference" in s: return "dpo_pairs"
        return "json_artifact"
    if ext in (".yaml",".yml",".toml",".cfg",".ini"): return "configuration"
    if ext in (".pt",".pth",".ckpt",".pkl",".pickle",".npz",".npy"): return "model_checkpoint"
    if ext in (".png",".jpg",".jpeg",".svg",".pdf"): return "figure"
    if ext in (".log",".out",".txt"): return "log"
    if ext in (".md",".docx",".tex",".rst"): return "documentation"
    if ext in (".sp",".cir",".net",".lib",".mod"): return "spice_data"
    if "/archive/" in s: return "archive"
    return "other"

def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

rows, summary = [], {}
targets = [ROOT/d for d in LEGACY_DIRS if (ROOT/d).exists()]
files = [ROOT/f for f in LEGACY_ROOT_FILES if (ROOT/f).is_file()]
with open(ROOT/"ORIGINAL_RAPTOR_FILE_INVENTORY.csv","w",newline="",encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["original_path","file_name","extension","file_size","category",
                "referenced_by","proposed_destination","content_hash","classification_confidence","notes"])
    def emit(p: Path, top: str):
        rel = p.relative_to(ROOT)
        cat = category(p)
        bulk = top in BULK
        try: size = p.stat().st_size
        except OSError: size = -1
        h = "skipped-bulk" if bulk or cat == "cache" else (md5(p) if size >= 0 and size < 50_000_000 else "skipped-large")
        w.writerow([str(rel), p.name, p.suffix, size, cat, top,
                    f"RAPTOR_Legacy/{rel}", h, "high" if not bulk else "high-bulk",
                    "generated artifact tree" if bulk else ""])
        summary[cat] = summary.get(cat,0)+1
    for f in files: emit(f, "(root)")
    for d in targets:
        for dirpath, _dirs, names in os.walk(d):
            for n in names: emit(Path(dirpath)/n, d.name)
json.dump({"root": str(ROOT), "total_files": sum(summary.values()),
           "by_category": summary, "legacy_dirs": [d.name for d in targets],
           "legacy_root_files": [f.name for f in files]},
          open(ROOT/"ORIGINAL_RAPTOR_FILE_INVENTORY.json","w",encoding="utf-8"), indent=1)
print("TOTAL:", sum(summary.values())); print(json.dumps(summary, indent=0))
