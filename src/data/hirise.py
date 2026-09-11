"""Read the public HiRISE labels without assigning anomaly-detection roles."""
from collections import Counter, defaultdict
import csv
from pathlib import Path


def load_hirise_metadata(root: Path) -> tuple[dict, dict]:
    label_files = sorted(root.rglob('labels-map-proj*.txt'))
    maps = sorted(p for p in root.rglob('*classmap.csv') if not p.name.startswith('._') and '__MACOSX' not in p.parts)
    if not label_files and not maps:
        return {}, {}
    primary = [p for p in label_files if 'train_val_test' not in p.name]
    supplied = [p for p in label_files if 'train_val_test' in p.name]
    if len(primary) != 1 or len(supplied) != 1 or len(maps) != 1:
        raise ValueError('Expected exactly one HiRISE label file, supplied split file and class map')
    classes = {}
    with maps[0].open(encoding='utf-8-sig', newline='') as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields:
                continue
            try:
                identifier = int(fields[0])
            except ValueError:
                if line == 1 and 'id' in fields[0].lower():
                    continue
                raise ValueError(f'Invalid class map row {line}: {fields}')
            if len(fields) != 2 or identifier in classes:
                raise ValueError(f'Invalid or duplicate class map row: {fields}')
            classes[identifier] = fields[1].strip()
    metadata = {}
    repeated_labels = 0
    for line, text in enumerate(primary[0].read_text().splitlines(), 1):
        if not text.strip():
            continue
        fields = text.split()
        if len(fields) != 2:
            raise ValueError(f'Invalid primary label row {line}')
        filename, class_id = Path(fields[0]).name, int(fields[1])
        if class_id not in classes:
            raise ValueError(f'Unknown class ID {class_id}')
        if filename in metadata:
            if metadata[filename]['class_id'] != class_id:
                raise ValueError(f'Conflicting primary labels: {filename}')
            repeated_labels += 1
        metadata[filename] = {'class_id': class_id, 'class_name': classes[class_id], 'supplied_splits': set(), 'supplied_row_count': 0, 'supplied_split_row_counts': Counter()}
    row_counts, split_files, mismatches = Counter(), defaultdict(set), []
    for line, text in enumerate(supplied[0].read_text().splitlines(), 1):
        if not text.strip():
            continue
        fields = text.split()
        if len(fields) != 3:
            raise ValueError(f'Invalid supplied split row {line}')
        filename, class_id, split = Path(fields[0]).name, int(fields[1]), fields[2].lower()
        split = {'validation': 'val', 'valid': 'val', 'training': 'train', 'testing': 'test'}.get(split, split)
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown supplied split {split!r}')
        if filename not in metadata or metadata[filename]['class_id'] != class_id:
            mismatches.append({'line': line, 'filename': filename, 'class_id': class_id})
            continue
        metadata[filename]['supplied_splits'].add(split)
        metadata[filename]['supplied_row_count'] += 1
        metadata[filename]['supplied_split_row_counts'][split] += 1
        row_counts[split] += 1
        split_files[split].add(filename)
    for entry in metadata.values():
        entry['supplied_splits'] = sorted(entry['supplied_splits'])
    report = {'class_map': classes, 'primary_label_rows': len(metadata) + repeated_labels,
              'primary_unique_filenames': len(metadata), 'repeated_primary_label_rows': repeated_labels,
              'supplied_split_rows': dict(row_counts),
              'supplied_split_unique_images': {key: len(value) for key, value in split_files.items()},
              'supplied_label_mismatches': mismatches,
              'images_absent_from_supplied_split': [name for name, entry in metadata.items() if not entry['supplied_splits']],
              'same_filename_in_multiple_splits': [name for name, entry in metadata.items() if len(entry['supplied_splits']) > 1],
              'repeated_supplied_rows': sum(max(0, entry['supplied_row_count'] - 1) for entry in metadata.values()),
              'metadata_files': [p.relative_to(root).as_posix() for p in [primary[0], supplied[0], maps[0]]],
              'normal_anomaly_roles': 'UNASSIGNED: semantic classes are not anomaly labels.'}
    return metadata, report
