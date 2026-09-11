"""Verify the requested archive before safely extracting it."""
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/raw/hirise-map-proj-v3_2.zip'
DESTINATION = ROOT / 'data/raw/hirise_v3_2'
EXPECTED = '236d9c627db1a5970e77a01a8c8a035a'


def main():
    with ARCHIVE.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'md5').hexdigest()
    if actual != EXPECTED:
        raise ValueError(f'MD5 mismatch: {actual}')
    print(f'MD5 verified: {actual}', flush=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        targets = []
        for member in members:
            parts = PurePosixPath(member.filename).parts
            # Remove the archive's packaging folder, preserving the actual dataset hierarchy.
            if parts and parts[0] == 'hirise-map-proj-v3_2':
                parts = parts[1:]
            target = DESTINATION.joinpath(*parts).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Archive symlink rejected: {member.filename}')
            targets.append(target)
        for index, (member, target) in enumerate(zip(members, targets), 1):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
            if index % 10000 == 0:
                print(f'Extracted {index:,}/{len(members):,} entries', flush=True)
    report = {'record': 'https://zenodo.org/records/4002935',
              'download_url': 'https://zenodo.org/records/4002935/files/hirise-map-proj-v3_2.zip?download=1',
              'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
              'md5': actual, 'md5_verified': True, 'extracted_to': str(DESTINATION)}
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'outputs/download_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
