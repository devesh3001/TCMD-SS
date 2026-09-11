"""Recursive image inventory and conservative connected-component leakage groups."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from src.data.split import grouped_split, source_safe_groups, class_covered_split
from src.data.hirise import load_hirise_metadata

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm', '.pgm'}
EXCLUDED = {'.venv', '.git', '__pycache__', '.pytest_cache', '.pytest-tmp', 'outputs'}
SPLITS = {'train': 'train', 'training': 'train', 'test': 'test', 'testing': 'test',
          'val': 'val', 'valid': 'val', 'validation': 'val'}
AUGMENT = re.compile(r'(?:[_\-.](?:r(?:90|180|270)|rot(?:ate)?[_-]?(?:90|180|270)|fh|fv|hflip|vflip|brt|brightness(?:[_-]?[0-9.]+)?|flip[_-]?[hv]))$', re.I)
COLUMNS = ['path', 'status', 'error', 'format', 'width', 'height', 'mode', 'channels', 'frames',
           'label', 'class_name', 'original_split', 'base_id', 'augmentations', 'sha256',
           'pixel_sha256', 'phash', 'mean', 'std', 'group_id', 'flags', 'class_id', 'source_id', 'supplied_splits', 'supplied_row_count', 'supplied_split_row_counts']


def files_under(root: Path):
    # Do not follow symlinks: cycles and external data must not silently expand scope.
    for directory, dirs, files in root.walk(follow_symlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (directory / d).is_symlink())
        for name in sorted(files):
            path = directory / name
            if not path.is_symlink():
                yield path


def detect_candidates(workspace: Path) -> list[Path]:
    candidates = set()
    for path in files_under(workspace):
        if path.suffix.lower() in EXTENSIONS:
            relative = path.relative_to(workspace)
            candidates.add(workspace / relative.parts[0] if len(relative.parts) > 1 else workspace)
    return sorted(candidates)


def base_identifier(stem: str) -> tuple[str, list[str]]:
    transformations = []
    while match := AUGMENT.search(stem):
        transformations.append(match.group()[1:].lower())
        stem = stem[:match.start()]
    return stem.casefold(), list(reversed(transformations))


def audit(root: Path, normal_labels: list[str], anomaly_labels: list[str], threshold: int = 4):
    if not 0 <= threshold <= 64:
        raise ValueError('phash_distance must be between 0 and 64')
    rows, unusual = [], []
    metadata, metadata_report = load_hirise_metadata(root)
    observed_names = set()
    folders = {'.'}
    normal, anomaly = set(map(str.casefold, normal_labels)), set(map(str.casefold, anomaly_labels))
    for path in files_under(root):
        relative = path.relative_to(root)
        folders.update(parent.as_posix() for parent in relative.parents)
        parts = [p.casefold() for p in relative.parts[:-1]]
        labels = ({'normal'} if normal.intersection(parts) else set()) | ({'anomaly'} if anomaly.intersection(parts) else set())
        label = next(iter(labels)) if len(labels) == 1 else 'unknown'
        splits = set(SPLITS[p] for p in parts if p in SPLITS)
        base, aug = base_identifier(path.stem)
        row = dict.fromkeys(COLUMNS, None)
        source = re.fullmatch(r'((?:esp|psp|tra)_\d{6}_\d{4})_red-\d+', base)
        row['source_id'] = source[1] if source else None
        row.update(path=relative.as_posix(), status='ok', label=label,
                   class_name='/'.join(p for p in parts if p not in SPLITS),
                   original_split=next(iter(splits)) if len(splits) == 1 else 'unknown',
                   base_id=base, augmentations=aug, flags=[])
        if len(labels) > 1:
            row['flags'].append('conflicting_folder_labels')
        if len(splits) > 1:
            row['flags'].append('conflicting_folder_splits')
        if path.name.startswith('._'):
            with path.open('rb') as stream:
                is_resource_fork = stream.read(4) == bytes.fromhex('00051607')
            if is_resource_fork:
                row.update(status='auxiliary', error='AppleDouble macOS resource-fork sidecar')
                rows.append(row)
                continue
        if path.name in metadata:
            entry = metadata[path.name]
            observed_names.add(path.name)
            row.update(entry)
            # Semantic terrain classes remain independent of normal/anomaly roles.
            row['label'] = 'unknown'
            row['original_split'] = '|'.join(entry['supplied_splits']) or 'unknown'
        elif metadata and path.suffix.lower() in EXTENSIONS:
            row['flags'].append('missing_primary_label')
        if metadata and row['path'] in metadata_report['metadata_files']:
            row['status'] = 'metadata'
            rows.append(row)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                row.update(format=image.format, width=image.width, height=image.height,
                           mode=image.mode, channels=len(image.getbands()), frames=getattr(image, 'n_frames', 1))
                rgb = image.convert('RGB')
                row['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                row['phash'] = str(imagehash.phash(rgb))
                stats = ImageStat.Stat(rgb.convert('L'))
                row.update(mean=stats.mean[0], std=stats.stddev[0])
                if row['frames'] > 1:
                    row['flags'].append('multiple_frames_hashes_cover_first_frame_only')
                if row['std'] < 1:
                    row['flags'].append('nearly_constant_image')
            with path.open('rb') as stream:
                row['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            if path.suffix.lower() not in EXTENSIONS:
                row['flags'].append('unexpected_image_extension')
        except Exception as error:
            row.update(status='corrupt' if path.suffix.lower() in EXTENSIONS else 'unsupported', error=str(error))
            unusual.append({'path': row['path'], 'status': row['status'], 'error': str(error)})
        rows.append(row)
        if len(rows) % 5000 == 0:
            print(f'Inspected {len(rows):,} files', flush=True)
    valid = [i for i, row in enumerate(rows) if row['status'] == 'ok']
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    def connect(key):
        buckets = defaultdict(list)
        for i in valid:
            buckets[rows[i][key]].append(i)
        groups = [members for members in buckets.values() if len(members) > 1]
        for members in groups:
            for i in members[1:]:
                union(members[0], i)
        return [[rows[i]['path'] for i in members] for members in groups]

    # Global base IDs may over-group repeated generic names, but never separate named variants.
    print(f'Grouping {len(valid):,} valid images', flush=True)
    base_groups = connect('base_id')
    exact = connect('sha256')
    pixel = connect('pixel_sha256')
    near = []
    # With distance <= t, at least one of t+1 disjoint bit blocks is identical.
    # This exact candidate index avoids an all-pairs scan without losing matches.
    hashes = {i: int(rows[i]['phash'], 16) for i in valid}
    index = defaultdict(list)
    blocks = min(threshold + 1, 64)
    boundaries = np.linspace(0, 64, blocks + 1, dtype=int)
    for a in valid:
        keys = [(block, (hashes[a] >> int(boundaries[block])) & ((1 << int(boundaries[block + 1] - boundaries[block])) - 1)) for block in range(blocks)]
        candidates = set(range(a)) if threshold == 64 else {b for key in keys for b in index[key]}
        for b in candidates:
            if b not in hashes:
                continue
            distance = (hashes[a] ^ hashes[b]).bit_count()
            if distance <= threshold:
                union(a, b)
                if rows[a]['pixel_sha256'] != rows[b]['pixel_sha256']:
                    near.append({'a': rows[a]['path'], 'b': rows[b]['path'], 'distance': distance})
        for key in keys:
            index[key].append(a)

    groups = defaultdict(list)
    for i in valid:
        groups[find(i)].append(i)
    leakage, conflicts = [], []
    for members in groups.values():
        group_id = hashlib.sha256('\n'.join(sorted(rows[i]['path'] for i in members)).encode()).hexdigest()[:16]
        for i in members:
            rows[i]['group_id'] = group_id
        known_splits = {split for i in members for split in rows[i]['original_split'].split('|')} - {'unknown'}
        known_labels = {rows[i]['class_id'] if metadata else rows[i]['label'] for i in members} - {'unknown', None}
        if len(known_splits) > 1:
            leakage.append({'group_id': group_id, 'splits': sorted(known_splits), 'paths': [rows[i]['path'] for i in members]})
        if len(known_labels) > 1:
            conflicts.append(group_id)
            for i in members:
                rows[i]['flags'].append('group_label_conflict')
    normal_train = [i for i in valid if rows[i]['label'] == 'normal' and rows[i]['original_split'] == 'train']
    # Brightness outliers are review hints only, never inferred anomaly ground truth.
    if len(normal_train) >= 5:
        values = np.array([rows[i]['mean'] for i in normal_train])
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        for i in normal_train:
            if abs(rows[i]['mean'] - median) > max(20, 4.5 * 1.4826 * mad):
                rows[i]['flags'].append('normal_train_brightness_outlier')
    suspicious = [rows[i]['path'] for i in valid if rows[i]['original_split'] == 'train' and
                  (rows[i]['label'] == 'anomaly' or (rows[i]['label'] == 'normal' and rows[i]['flags']))]
    augmented_paths = {rows[i]['path'] for i in valid if rows[i]['augmentations']}
    summary = {
        'status': 'completed', 'selected_dataset': str(root),
        'files_scanned': len(rows), 'image_count': len(valid),
        'file_status_counts': dict(Counter(r['status'] for r in rows)),
        'normal_images': sum(rows[i]['label'] == 'normal' for i in valid),
        'anomaly_images': sum(rows[i]['label'] == 'anomaly' for i in valid),
        'unknown_label_images': sum(rows[i]['label'] == 'unknown' for i in valid),
        'folder_structure': sorted(folders),
        'dimensions': dict(Counter(f"{rows[i]['width']}x{rows[i]['height']}" for i in valid)),
        'channels': dict(Counter(str(rows[i]['channels']) for i in valid)),
        'formats': dict(Counter(rows[i]['format'] for i in valid)),
        'classes': dict(Counter(rows[i]['class_name'] for i in valid)),
        'original_splits': dict(Counter(rows[i]['original_split'] for i in valid)),
        'label_counts_by_split': {split: dict(Counter(rows[i]['label'] for i in valid if rows[i]['original_split'] == split)) for split in sorted({rows[i]['original_split'] for i in valid})},
        'naming_conventions': {'augmentation_tokens': dict(Counter(token for i in valid for token in rows[i]['augmentations'])), 'examples': [{'filename': rows[i]['path'], 'base_id': rows[i]['base_id']} for i in valid[:20]]},
        'class_imbalance': {'label_counts': dict(Counter(rows[i]['label'] for i in valid)), 'note': 'Compare known labels within each split; unknown labels cannot establish class balance.'},
        'augmentation_images': sum(bool(rows[i]['augmentations']) for i in valid),
        'augmentation_groups': [g for g in base_groups if augmented_paths.intersection(g)],
        'augmentation_group_size_distribution': dict(Counter(str(size) for size in Counter(rows[i]['base_id'] for i in valid).values())),
        'base_groups': base_groups, 'exact_duplicate_groups': exact, 'pixel_duplicate_groups': pixel,
        'near_duplicate_pairs': near, 'group_count': len(groups), 'possible_leakage': leakage,
        'label_conflict_groups': conflicts, 'unusual_files': unusual, 'possible_training_anomalies': suspicious,
        'recommended_safe_split': '70/15/15 by connected group; review labels and flagged samples, use only confirmed normal training groups; keep every group wholly in one split.',
        'limitations': ['Folder labels require explicit vocabulary; unknown labels are not normal.',
                       'Filename inference recognizes terminal delimited rotation, flip and brightness tokens.',
                       'Global repeated base names and pHash matches conservatively over-group.',
                       'pHash is heuristic and is not rotation invariant; unnamed transformations may be missed.',
                       'No semantic anomaly detector is used; training flags require manual review.']}
    if metadata:
        summary['hirise_metadata'] = metadata_report
        source_splits = defaultdict(set)
        for i in valid:
            if rows[i]['source_id']:
                source_splits[rows[i]['source_id']].update(rows[i]['supplied_splits'] or [])
        summary['source_observation_count'] = len(source_splits)
        summary['unparsed_source_ids'] = [rows[i]['path'] for i in valid if not rows[i]['source_id']]
        summary['supplied_source_overlap'] = {key: sorted(value) for key, value in source_splits.items() if len(value) > 1}
        summary['labels_missing_images'] = sorted(set(metadata) - observed_names)
        summary['images_missing_labels'] = [rows[i]['path'] for i in valid if rows[i]['class_id'] is None]
        summary['normal_images'] = None
        summary['anomaly_images'] = None
        summary['possible_training_anomalies'] = []
        summary['training_anomaly_review_status'] = 'Not assessable until normal classes are designated; low-level quality flags remain available.'
    return pd.DataFrame(rows, columns=COLUMNS), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path, default=WORKSPACE / 'configs/base.yaml')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = args.output_dir or WORKSPACE / config['audit']['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    candidates = detect_candidates(WORKSPACE)
    selected = args.dataset or config['dataset']['root']
    if selected is None and len(candidates) == 1:
        selected = candidates[0]
    print('Dataset candidates:', [str(p) for p in candidates])
    if selected is None:
        summary = {'status': 'dataset_missing' if not candidates else 'dataset_ambiguous',
                   'candidate_directories': list(map(str, candidates)), 'selected_dataset': None,
                   'message': 'Provide --dataset PATH. No real dataset was audited.'}
        pd.DataFrame(columns=COLUMNS).to_csv(output / 'dataset_audit.csv', index=False)
        (output / 'dataset_audit.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 2
    root = Path(selected).expanduser()
    root = (WORKSPACE / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir():
        parser.error(f'Dataset directory does not exist: {root}')
    print('Selected dataset:', root)
    frame, summary = audit(root, config['dataset']['normal_labels'], config['dataset']['anomaly_labels'], config['audit']['phash_distance'])
    summary['candidate_directories'] = list(map(str, candidates))
    summary['seed'] = config['split']['seed']
    try:
        candidates_frame = frame.loc[frame.status.eq('ok')].copy()
        if not candidates_frame.empty and candidates_frame.source_id.notna().all():
            candidates_frame = source_safe_groups(candidates_frame)
            summary['source_safe_split_group_count'] = candidates_frame.split_group_id.nunique()
        split = class_covered_split(candidates_frame, tuple(config['split']['ratios']), config['split']['seed'])
        # Record proposals in the same manifest to avoid stale standalone split files.
        split_columns = ['path', 'assigned_split'] + (['split_group_id'] if 'split_group_id' in split else [])
        frame = frame.merge(split[split_columns], on='path', how='left')
        summary['proposed_split_counts'] = split.assigned_split.value_counts().to_dict()
        summary['allocation_seed'] = split.attrs.get('allocation_seed', config['split']['seed'])
        summary['split_selection'] = 'First complete-class-coverage allocation from a fixed seed-42 sequence; full source and duplicate groups remain intact.'
    except ValueError as error:
        summary['split_unavailable'] = str(error)
    valid_frame = frame.loc[frame.status.eq('ok')].copy()
    distributions = []
    if 'hirise_metadata' in summary:
        for class_id, class_name in summary['hirise_metadata']['class_map'].items():
            subset = valid_frame.loc[valid_frame.class_id.eq(class_id)]
            record = {'class_id': class_id, 'class_name': class_name, 'image_count': len(subset),
                      'original_image_count': int(subset.augmentations.map(lambda x: not x).sum()),
                      'augmented_image_count': int(subset.augmentations.map(bool).sum()),
                      'base_landmark_count': subset.base_id.nunique()}
            for split_name in ['train', 'val', 'test']:
                mask = subset.supplied_splits.map(lambda x: split_name in (x or []))
                record[f'supplied_{split_name}_unique_images'] = int(mask.sum())
                record[f'supplied_{split_name}_rows'] = int(subset.supplied_split_row_counts.map(lambda x: (x or {}).get(split_name, 0)).sum())
            distributions.append(record)
    pd.DataFrame(distributions).to_csv(output / 'class_distribution.csv', index=False)
    base_records = []
    for base, subset in valid_frame.groupby('base_id', sort=True):
        base_records.append({'base_id': base, 'image_count': len(subset),
                             'original_image_count': int(subset.augmentations.map(lambda x: not x).sum()),
                             'augmented_image_count': int(subset.augmentations.map(bool).sum()),
                             'class_ids': '|'.join(str(int(c)) for c in sorted(subset.class_id.dropna().unique())),
                             'group_id': subset.group_id.iloc[0],
                             'split_group_id': subset.split_group_id.iloc[0] if 'split_group_id' in subset else None,
                             'assigned_split': subset.assigned_split.iloc[0] if 'assigned_split' in subset else None,
                             'supplied_splits': '|'.join(sorted({s for value in subset.original_split for s in value.split('|')})),
                             'paths': json.dumps(subset.path.tolist())})
    pd.DataFrame(base_records).to_csv(output / 'base_image_groups.csv', index=False)
    summary['class_distribution'] = distributions
    if distributions:
        counts = [r['image_count'] for r in distributions]
        summary['class_imbalance'] = {'majority_image_fraction': max(counts) / sum(counts),
                                    'largest_to_smallest_class_ratio': max(counts) / min(counts) if min(counts) else None}
    path_rows = valid_frame.set_index('path')
    near_counts = Counter()
    for pair in summary['near_duplicate_pairs']:
        a, b = path_rows.loc[pair['a']], path_rows.loc[pair['b']]
        near_counts['same_base' if a.base_id == b.base_id else 'different_base'] += 1
        if a.original_split != b.original_split:
            near_counts['cross_supplied_split_pairs'] += 1
    summary['near_duplicate_pair_counts'] = dict(near_counts)
    summary['supplied_base_split_overlap_count'] = sum(len(set(s for value in part.original_split for s in value.split('|')) - {'unknown'}) > 1 for _, part in valid_frame.groupby('base_id'))
    if 'assigned_split' in valid_frame:
        summary['proposed_class_distribution'] = {str(int(c)): part.assigned_split.value_counts().to_dict() for c, part in valid_frame.dropna(subset=['class_id']).groupby('class_id')}
        summary['proposed_original_image_counts'] = valid_frame.loc[valid_frame.augmentations.map(lambda x: not x)].assigned_split.value_counts().to_dict()
    summary['base_landmark_count'] = len(base_records)
    if 'assigned_split' in valid_frame and valid_frame.source_id.notna().any():
        overlaps = valid_frame.groupby('source_id').assigned_split.nunique()
        summary['provisional_source_overlap_count'] = int((overlaps > 1).sum())
    summary['provisional_allocation_only'] = True
    summary['base_groups_without_original'] = [r['base_id'] for r in base_records if r['original_image_count'] == 0]
    summary['recommended_safe_split'] = 'Do not activate an anomaly experiment before normal-class selection. Use seed 42 and complete source observations linked by duplicate components for a provisional 70/15/15 allocation; train only on selected normal groups, evaluate held-out original images, and preferably hold out complete HiRISE source observations.'
    frame.to_csv(output / 'dataset_audit.csv', index=False)
    (output / 'dataset_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    compact = {k: summary[k] for k in ['image_count', 'normal_images', 'anomaly_images', 'unknown_label_images', 'dimensions', 'recommended_safe_split']}
    compact.update({k: len(summary[k]) for k in ['augmentation_groups', 'exact_duplicate_groups', 'pixel_duplicate_groups', 'near_duplicate_pairs', 'possible_leakage']})
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
