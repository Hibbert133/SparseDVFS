"""Run only in the target container with /model writable during prepare."""
import hashlib
import json
from pathlib import Path
from .contracts import MODEL_ID, REVISION, verify_model


def main():
    from huggingface_hub import snapshot_download
    root = Path('/model')
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    files = {}
    for file in sorted(root.iterdir()):
        if file.name not in shards and file.suffix not in ('.json', '.txt', '.jinja'):
            continue
        if file.name == 'sparsedvfs_model_lock.json':
            continue
        if file.is_file():
            with file.open('rb') as source:
                sha = hashlib.file_digest(source, 'sha256').hexdigest()
            files[file.name] = dict(size=file.stat().st_size, sha256=sha)
    (root / 'sparsedvfs_model_lock.json').write_text(json.dumps(
        dict(model_id=MODEL_ID, revision=REVISION, files=files), indent=2) + '\n')
    print(json.dumps(verify_model(root, hashes=True), indent=2))


if __name__ == '__main__':
    main()
