"""Gera uma máscara rotulada única para o MedAtlas a partir das saídas do TotalSegmentator.

Entrada esperada:
- atlas-torax/tc-real/dados/torax.nii.gz
- artefatos temporários em tmp/seg-total/ e opcionalmente tmp/seg-lung-vessels/

Saída:
- atlas-torax/tc-real/dados/mascara_torax.nii.gz
- atlas-torax/tc-real/dados/rotulos_torax.json
- atlas-torax/tc-real/dados/segmentacao_info.json
"""
from __future__ import annotations
from pathlib import Path
import json
import SimpleITK as sitk

ROOT = Path('.')
CT = ROOT / 'atlas-torax' / 'tc-real' / 'dados' / 'torax.nii.gz'
TOTAL_DIR = ROOT / 'tmp' / 'seg-total'
LUNG_DIR = ROOT / 'tmp' / 'seg-lung-vessels'
OUT_MASK = ROOT / 'atlas-torax' / 'tc-real' / 'dados' / 'mascara_torax.nii.gz'
OUT_LABELS = ROOT / 'atlas-torax' / 'tc-real' / 'dados' / 'rotulos_torax.json'
OUT_INFO = ROOT / 'atlas-torax' / 'tc-real' / 'dados' / 'segmentacao_info.json'

# Estruturas abertas que a automação tenta realmente gerar.
STRUCTURES = [
    ('lung_upper_lobe_right', 'Lobo superior direito'),
    ('lung_middle_lobe_right', 'Lobo médio direito'),
    ('lung_lower_lobe_right', 'Lobo inferior direito'),
    ('lung_upper_lobe_left', 'Lobo superior esquerdo'),
    ('lung_lower_lobe_left', 'Lobo inferior esquerdo'),
    ('trachea', 'Traqueia'),
    ('esophagus', 'Esôfago'),
    ('heart', 'Coração'),
    ('aorta', 'Aorta torácica'),
    ('pulmonary_vein', 'Veias pulmonares'),
    ('superior_vena_cava', 'Veia cava superior'),
]

OPTIONAL_FROM_LUNG_TASK = [
    ('lung_airways', 'Árvore traqueobrônquica'),
    ('lung_arteries', 'Artérias pulmonares'),
    ('lung_veins', 'Veias pulmonares intrapulmonares'),
]

PLANNED = [
    'Língula', 'Átrio direito', 'Átrio esquerdo', 'Ventrículo direito',
    'Ventrículo esquerdo', 'Brônquio principal direito', 'Brônquio principal esquerdo'
]


def read_img(path: Path):
    return sitk.ReadImage(str(path))


def to_bool(img):
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


if not CT.exists():
    raise SystemExit('Arquivo base não encontrado: atlas-torax/tc-real/dados/torax.nii.gz')

ref = read_img(CT)
shape = ref.GetSize()
out = sitk.Image(ref.GetSize(), sitk.sitkUInt16)
out.CopyInformation(ref)
labels: dict[str, str] = {}
current_id = 1
added = []
missing = []

def place(mask_path: Path, label_name: str):
    global out, current_id
    if not mask_path.exists():
        return False, 'arquivo ausente'
    mask = to_bool(read_img(mask_path))
    if mask.GetSize() != shape:
        return False, f'dimensões incompatíveis: {mask.GetSize()} vs {shape}'
    stats = sitk.StatisticsImageFilter()
    stats.Execute(mask)
    if stats.GetSum() <= 0:
        return False, 'máscara vazia'
    out = sitk.Mask(out, mask == 0) + sitk.Cast(mask, sitk.sitkUInt16) * current_id
    labels[str(current_id)] = label_name
    added.append({'id': current_id, 'label': label_name, 'source': str(mask_path)})
    current_id += 1
    return True, 'ok'

for filename, label in STRUCTURES:
    ok, reason = place(TOTAL_DIR / f'{filename}.nii.gz', label)
    if not ok:
        missing.append({'label': label, 'source_name': filename, 'reason': reason})

for filename, label in OPTIONAL_FROM_LUNG_TASK:
    ok, reason = place(LUNG_DIR / f'{filename}.nii.gz', label)
    if not ok:
        missing.append({'label': label, 'source_name': filename, 'reason': reason})

if not labels:
    raise SystemExit('Nenhuma estrutura foi adicionada à máscara final.')

OUT_MASK.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(out, str(OUT_MASK), useCompression=True)
write_json(OUT_LABELS, labels)
write_json(OUT_INFO, {
    'ct': str(CT),
    'mask': str(OUT_MASK),
    'labels': str(OUT_LABELS),
    'available_labels': labels,
    'added': added,
    'missing_or_skipped': missing,
    'planned_future_labels': PLANNED,
    'note': 'As estruturas planejadas dependem de máscaras adicionais, algumas sob licença específica.'
})
print('Máscara criada com', len(labels), 'estruturas.')
print('Rótulos disponíveis:', ', '.join(labels.values()))
