from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.audit_dataset import audit, base_identifier, detect_candidates
from src.data.dataset import ImageDataset
from src.data.split import grouped_split
from src.data.transforms import image_transform
from src.utils.seed import seed_everything


def save(path: Path, seed: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.random.default_rng(seed).integers(0, 256, (32, 40, 3), dtype=np.uint8))
    image.save(path)
    return image


@pytest.mark.parametrize('name,base,count', [('part_r90_fh', 'part', 2), ('part-r180', 'part', 1), ('part.fv', 'part', 1), ('r90part', 'r90part', 0), ('part_rot270', 'part', 1), ('part-brt', 'part', 1)])
def test_base(name, base, count):
    identifier, tokens = base_identifier(name)
    assert identifier == base
    assert len(tokens) == count


def test_audit(tmp_path):
    original = save(tmp_path / 'train/normal/item.png')
    target = tmp_path / 'test/defect/item_r90.png'
    target.parent.mkdir(parents=True)
    original.rotate(90, expand=True).save(target)
    shutil.copyfile(tmp_path / 'train/normal/item.png', tmp_path / 'test/defect/copy.png')
    original.save(tmp_path / 'test/defect/encoded.bmp')
    pixels = np.array(original)
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels).save(tmp_path / 'test/defect/near.png')
    (tmp_path / 'broken.png').write_bytes(b'broken')
    (tmp_path / 'metadata.txt').write_text('information')
    frame, report = audit(tmp_path, ['normal'], ['defect'])
    assert report['image_count'] == 5
    assert report['normal_images'] == 1 and report['anomaly_images'] == 4
    assert len(report['exact_duplicate_groups']) == 1
    assert len(report['pixel_duplicate_groups']) == 1
    assert report['near_duplicate_pairs']
    assert report['possible_leakage']
    assert report['possible_training_anomalies']
    assert frame.loc[frame.status.eq('ok'), 'group_id'].nunique() == 1
    assert len(report['unusual_files']) == 2
    dataset = ImageDataset(frame, tmp_path, image_transform((16, 16)))
    assert len(dataset) == 5
    assert dataset[0]['image'].shape == (3, 16, 16)


def test_split():
    frame = pd.DataFrame({'group_id': [f'g{i}' for i in range(20) for _ in range(3)], 'status': 'ok'})
    result = grouped_split(frame)
    assert result.groupby('group_id').assigned_split.nunique().max() == 1
    assert set(result.assigned_split) == {'train', 'val', 'test'}
    pd.testing.assert_frame_equal(result, grouped_split(frame))
    with pytest.raises(ValueError):
        grouped_split(frame.iloc[:3])
    with pytest.raises(ValueError):
        grouped_split(frame, (0.8, 0.2, 0.2))


def test_detection_and_unknown(tmp_path):
    assert detect_candidates(tmp_path) == []
    save(tmp_path / 'data/unlabeled/a.png')
    assert detect_candidates(tmp_path) == [tmp_path / 'data']
    frame, report = audit(tmp_path / 'data', ['normal'], ['defect'])
    assert report['unknown_label_images'] == 1
    assert frame.iloc[0]['label'] == 'unknown'


def test_seed():
    seed_everything()
    first = np.random.rand(3)
    seed_everything()
    assert np.array_equal(first, np.random.rand(3))


def test_unsupervised_requires_explicit_normal_classes():
    from src.data.split import unsupervised_split
    frame = pd.DataFrame({'group_id': [f'g{i}' for i in range(20) for _ in range(2)],
                          'status': 'ok', 'class_id': [0] * 20 + [1] * 20})
    with pytest.raises(ValueError):
        unsupervised_split(frame, set())
    result = unsupervised_split(frame, {0})
    assert result.loc[result.assigned_split.eq('train'), 'class_id'].eq(0).all()
    assert result.groupby('group_id').assigned_split.nunique().max() == 1


def test_hirise_metadata(tmp_path, monkeypatch):
    from src.data.hirise import load_hirise_metadata
    (tmp_path / 'landmarks_map-proj-v3_2_classmap.csv').write_text('0,other\n1,crater\n')
    (tmp_path / 'labels-map-proj-v3_2.txt').write_text('a.png 0\na_r90.png 0\nb.png 1\n')
    (tmp_path / 'labels-map-proj_v3_2_train_val_test.txt').write_text('a.png 0 train\na.png 0 train\na_r90.png 0 test\nb.png 1 val\n')
    metadata, report = load_hirise_metadata(tmp_path)
    assert metadata['a.png']['supplied_row_count'] == 2
    assert report['supplied_split_rows']['train'] == 2
    assert report['supplied_split_unique_images']['train'] == 1
    save(tmp_path / 'map-proj-v3_2/a.png')
    save(tmp_path / 'map-proj-v3_2/a_r90.png')
    save(tmp_path / 'map-proj-v3_2/b.png', seed=4)
    frame, report = audit(tmp_path, ['other'], ['crater'])
    assert report['normal_images'] is None and report['anomaly_images'] is None
    assert report['possible_leakage']
    assert report['labels_missing_images'] == []
    assert report['images_missing_labels'] == []
    assert frame.loc[frame.status.eq('ok'), 'label'].eq('unknown').all()

    import json
    from scripts.audit_dataset import main
    monkeypatch.setattr('sys.argv', ['audit_dataset.py', '--dataset', str(tmp_path), '--output-dir', str(tmp_path / 'outputs')])
    assert main() == 0
    stored = json.loads((tmp_path / 'outputs/dataset_audit.json').read_text())
    assert stored['class_distribution'][0]['supplied_train_rows'] == 2
    assert (tmp_path / 'outputs/base_image_groups.csv').is_file()
    assert (tmp_path / 'outputs/dataset_audit.csv').is_file()


def test_source_groups_preserve_transitive_duplicates():
    from src.data.split import source_safe_groups
    frame = pd.DataFrame({'group_id': ['a', 'a', 'b', 'c'] + [f'g{i}' for i in range(20)],
                          'source_id': ['s1', 's2', 's2', 's3'] + [f's{i+4}' for i in range(20)],
                          'status': 'ok'})
    linked = source_safe_groups(frame)
    assert linked.split_group_id.iloc[0] == linked.split_group_id.iloc[2]
    split = grouped_split(linked)
    assert split.groupby('source_id').assigned_split.nunique().max() == 1
    assert split.groupby('group_id').assigned_split.nunique().max() == 1


def test_appledouble_is_not_a_corrupt_image(tmp_path):
    (tmp_path / '._a.jpg').write_bytes(bytes.fromhex('00051607') + b'archive metadata')
    frame, report = audit(tmp_path, [], [])
    assert report['image_count'] == 0
    assert report['file_status_counts']['auxiliary'] == 1
    assert frame.iloc[0].status == 'auxiliary'


@pytest.mark.parametrize('threshold', [0, 4, 64])
def test_indexed_near_duplicates_match_exhaustive_check(tmp_path, monkeypatch, threshold):
    import imagehash
    hashes = [0, 0, 15, 31, 1 << 63, (1 << 64) - 1]
    for i in range(len(hashes)):
        save(tmp_path / f'image{i}.png', seed=i)
    values = iter(hashes)
    monkeypatch.setattr('scripts.audit_dataset.imagehash.phash', lambda image: imagehash.hex_to_hash(f'{next(values):016x}'))
    _, report = audit(tmp_path, [], [], threshold)
    actual = {frozenset((pair['a'], pair['b'])) for pair in report['near_duplicate_pairs']}
    expected = {frozenset((f'image{i}.png', f'image{j}.png'))
                for i in range(len(hashes)) for j in range(i)
                if (hashes[i] ^ hashes[j]).bit_count() <= threshold}
    assert actual == expected


def test_class_coverage_split_is_deterministic():
    from src.data.split import class_covered_split
    frame = pd.DataFrame({'group_id': [f'g{i}' for i in range(30)],
                          'status': 'ok', 'class_id': [i % 3 for i in range(30)]})
    first = class_covered_split(frame)
    pd.testing.assert_frame_equal(first, class_covered_split(frame))
    assert all(set(part.class_id) == {0, 1, 2} for _, part in first.groupby('assigned_split'))
