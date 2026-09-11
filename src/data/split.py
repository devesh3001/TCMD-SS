import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(manifest: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 42) -> pd.DataFrame:
    """Assign complete groups; ratios target groups, not image counts or class balance."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('Provide three positive ratios summing to one')
    group_column = 'split_group_id' if 'split_group_id' in manifest else 'group_id'
    if manifest.empty or manifest['group_id'].isna().any() or manifest[group_column].isna().any():
        raise ValueError('A nonempty manifest with complete group IDs is required')
    if not manifest['status'].eq('ok').all():
        raise ValueError('Exclude corrupt and unsupported files before splitting')
    if manifest.groupby('group_id')[group_column].nunique().max() != 1:
        raise ValueError('A base/duplicate component cannot span splitting groups')
    if manifest[group_column].nunique() < 3:
        raise ValueError('At least three independent groups are required')
    result = manifest.copy().reset_index(drop=True)
    first = GroupShuffleSplit(n_splits=1, train_size=ratios[0], random_state=seed)
    train, holdout = next(first.split(result, groups=result[group_column]))
    remainder = result.iloc[holdout]
    if remainder[group_column].nunique() < 2:
        raise ValueError('Ratios leave fewer than two holdout groups; adjust ratios or add groups')
    second = GroupShuffleSplit(n_splits=1, train_size=ratios[1] / (ratios[1] + ratios[2]), random_state=seed)
    val, test = next(second.split(remainder, groups=remainder[group_column]))
    result['assigned_split'] = 'train'
    result.loc[holdout[val], 'assigned_split'] = 'val'
    result.loc[holdout[test], 'assigned_split'] = 'test'
    return result



def unsupervised_split(manifest: pd.DataFrame, normal_class_ids: set[int],
                       ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                       seed: int = 42) -> pd.DataFrame:
    """Require an explicit research choice; no semantic class is implicitly normal."""
    if not normal_class_ids:
        raise ValueError('Explicit normal_class_ids are required')
    if not normal_class_ids.issubset(set(manifest['class_id'].dropna())):
        raise ValueError('Normal class IDs must exist in the manifest')
    result = grouped_split(manifest, ratios, seed)
    eligible = result['class_id'].isin(normal_class_ids)
    group_column = 'split_group_id' if 'split_group_id' in result else 'group_id'
    unsafe_groups = set(result.loc[~eligible, group_column])
    # Move entire mixed/unknown/anomaly groups, never individual augmented rows.
    move = result[group_column].isin(unsafe_groups) & result.assigned_split.eq('train')
    result.loc[move, 'assigned_split'] = 'test'
    result['anomaly_label'] = result['class_id'].map(
        lambda value: -1 if pd.isna(value) else int(value not in normal_class_ids))
    if not result.assigned_split.eq('train').any():
        raise ValueError('No safe normal training groups remain; revise the explicit experiment')
    return result


def source_safe_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate components sharing a source observation to prevent scene overlap."""
    if manifest.empty or manifest[['source_id', 'group_id']].isna().any().any():
        raise ValueError('Complete source IDs and duplicate groups are required')
    parents = {group: group for group in manifest.group_id.unique()}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for _, subset in manifest.groupby('source_id', sort=True):
        groups = sorted(subset.group_id.unique())
        for group in groups[1:]:
            parents[find(group)] = find(groups[0])
    components = {}
    for group in sorted(parents):
        components.setdefault(find(group), []).append(group)
    identifiers = {root: hashlib.sha256('\n'.join(groups).encode()).hexdigest()[:16]
                   for root, groups in components.items()}
    result = manifest.copy()
    result['split_group_id'] = result.group_id.map(lambda group: identifiers[find(group)])
    return result


def class_covered_split(manifest: pd.DataFrame,
                        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                        seed: int = 42, attempts: int = 128) -> pd.DataFrame:
    """Find a reproducible group allocation covering each semantic class in each split."""
    classes = set(manifest['class_id'].dropna())
    if not classes:
        return grouped_split(manifest, ratios, seed)
    # Selection uses only class coverage, never downstream model performance.
    for attempt in range(attempts):
        result = grouped_split(manifest, ratios, seed + attempt)
        if all(set(part.class_id.dropna()) == classes for _, part in result.groupby('assigned_split')):
            result.attrs['allocation_seed'] = seed + attempt
            return result
    raise ValueError('No class-covered group split found; revise ratios or define the experiment explicitly')
