"""Baixa exames já obtidos pelo workflow e constrói par NIfTI alinhado.
Uso educacional; não produz segmentações novas e não é válido para diagnóstico.
"""
from pathlib import Path
import json, os, sys
import numpy as np
import nibabel as nib

src = Path(os.environ.get('SOURCE_DATA', '/tmp/i4h-data/patient-digital-twin/examples/data/s0011'))
out = Path(__file__).resolve().parents[1] / 'atlas-radiologia' / 'dados'
out.mkdir(parents=True, exist_ok=True)
ctpath = src / 'ct.nii.gz'
if not ctpath.exists(): raise RuntimeError('Arquivo CT não encontrado: '+str(ctpath))
if ctpath.stat().st_size < 100_000: raise RuntimeError('O arquivo TC parece um ponteiro Git LFS, não a TC real')
ct = nib.load(str(ctpath))
if len(ct.shape)!=3: raise RuntimeError('Tomografia não possui 3 dimensões')
# Cortes de 2 voxels para diminuir memória e permitir download pelo iPhone.
step = (2, 2, 2) if max(ct.shape)>300 else (1,1,2)
slices=tuple(slice(None,None,s) for s in step)
data = np.asarray(ct.dataobj[slices], dtype=np.float32)
if not np.isfinite(data).all(): raise RuntimeError('TC contém valores inválidos')
data = np.clip(np.rint(data),-2048,3071).astype(np.int16)
affine = np.array(ct.affine,dtype=float); affine[:3,:3]=affine[:3,:3] @ np.diag(step)
nii=nib.Nifti1Image(data,affine)
nii.header.set_xyzt_units('mm')
ctoutput=out/'tc.nii.gz'; nib.save(nii,str(ctoutput))
labels=[
 ('aorta','Aorta abdominal'),('liver','Fígado'),('spleen','Baço'),
 ('kidney_right','Rim direito'),('kidney_left','Rim esquerdo'),
 ('pancreas','Pâncreas'),('stomach','Estômago'),('gallbladder','Vesícula biliar'),
 ('inferior_vena_cava','Veia cava inferior'),('colon','Cólon'),
 ('small_bowel','Intestino delgado')]
mask=np.zeros(data.shape,dtype=np.uint8)
found={}
for name,label in labels:
 p=src/'segmentations'/f'{name}.nii.gz'
 if not p.exists():
  print('Sem máscara:',name);continue
 m=nib.load(str(p))
 if m.shape != ct.shape or not np.allclose(m.affine, ct.affine, atol=.075):
  raise RuntimeError(f'Máscara com dimensões ou geometria incompatível: {name}')
 values=np.asarray(m.dataobj[slices])
 ispart=values>.5
 if not np.any(ispart):
  print('Máscara sem voxels:',name);continue
 idx=len(found)+1
 mask[(ispart)&(mask==0)] = idx
 found[idx]=label
 print('Incluída:',name,'voxels=',int(ispart.sum()))
if not found: raise RuntimeError('Nenhuma máscara válida encontrada: verifique os dados s0011')
seg=nib.Nifti1Image(mask,affine)
seg.header.set_xyzt_units('mm')
nib.save(seg,str(out/'rotulos.nii.gz'))
meta={'dataset':'TotalSegmentator small s0011','source':'NVIDIA Isaac for Healthcare / TotalSegmentator',
      'license':'CC BY 4.0','size':list(map(int,data.shape)), 'labels':found,
      'files':{'ct':'dados/tc.nii.gz','seg':'dados/rotulos.nii.gz'},
      'disclaimer':'Rótulos do conjunto de dados, não validação clínica. Os ramos do cólon não estão separados.'}
(out/'manifesto.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('Pacote pronto!',ctoutput.stat().st_size,(out/'rotulos.nii.gz').stat().st_size,'bytes; regiões=',len(found))
