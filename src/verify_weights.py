import hashlib
import os

MANIFEST = "results/phase4_weights.sha256"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_phase4_weights(targets=None):
    """Verify gitignored .pt/.onnx weights against results/phase4_weights.sha256."""
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(f"Missing checksum manifest: {MANIFEST}")

    expected = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            digest = parts[0].upper()
            rel = os.path.normpath(parts[1].strip())
            expected[rel] = digest

    check_list = [os.path.normpath(t) for t in targets] if targets else list(expected.keys())
    for rel in check_list:
        if rel not in expected:
            raise KeyError(f"{rel} not found in {MANIFEST}")
        if not os.path.exists(rel):
            raise FileNotFoundError(f"Expected weight file missing: {rel}")
        actual = sha256_file(rel)
        if actual != expected[rel]:
            raise RuntimeError(
                f"WEIGHT DRIFT DETECTED in {rel}!\n"
                f"  expected: {expected[rel]}\n"
                f"  actual  : {actual}"
            )
    print(f"[OK] Verified {len(check_list)} weight file(s) against {MANIFEST}")


if __name__ == "__main__":
    verify_phase4_weights()