#!/usr/bin/env python3
import json,math,re
from pathlib import Path
from collections import defaultdict
import numpy as np
PHI=(1+math.sqrt(5))/2
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt')
records=[]; symbols=[]
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if not m or '.' not in m.group(1): continue
 locus=m.group(1); text=m.group(2); chars=re.findall(r'[A-Za-z]',re.sub(r'<->|\{[^}]*\}|\[[^]]*\]|@\d+','',text))
 if not chars: continue
 page=locus.split('.')[0]; linepart=locus.split('.')[1].split(',')[0] if '.' in locus else '0';
 records.append({'locus':locus,'page':page,'line':int(linepart) if linepart.isdigit() else 0,'n':len(chars),'start':len(symbols)}); symbols.extend(chars)
alphabet=sorted(set(symbols)); idx={c:i for i,c in enumerate(alphabet)}; X=np.zeros((len(alphabet),len(symbols)))
for n,c in enumerate(symbols): X[idx[c],n]=1
# phi-transition prediction and residual.
pred=PHI**-1*np.roll(X,1,axis=1)+PHI**-2*np.roll(X,-1,axis=1); E=X-pred; energy=np.sum(E*E,axis=0)
for r in records:
 e=energy[r['start']:r['start']+r['n']]; r['residual_energy']=float(e.sum()); r['residual_per_symbol']=float(e.mean()); r['symbol_count']=r['n']
page=defaultdict(lambda:[0,0,0]);
for r in records: page[r['page']][0]+=r['residual_energy']; page[r['page']][1]+=r['symbol_count']; page[r['page']][2]+=1
page_rows=[{'folio':p,'residual_energy':v[0],'symbols':v[1],'lines':v[2],'energy_per_symbol':v[0]/v[1]} for p,v in page.items()]
page_rows.sort(key=lambda x:x['residual_energy'],reverse=True)
# layout bands: normalize line position within each folio and aggregate into 20 bins.
bands=np.zeros(20); counts=np.zeros(20)
for r in records:
 pos=(r['line']-1)/max(1,40-1); b=min(19,max(0,int(pos*20))); bands[b]+=r['residual_energy']; counts[b]+=r['symbol_count']
band_rows=[{'layout_band':i,'residual_energy':float(bands[i]),'symbols':int(counts[i]),'energy_per_symbol':float(bands[i]/counts[i]) if counts[i] else 0} for i in range(20)]
res={'source':str(src),'phi':PHI,'operator':'prediction=phi^-1 previous + phi^-2 next','symbols':len(symbols),'folios':len(page_rows),'records':len(records),'alphabet':alphabet,'folio_residuals':page_rows,'layout_bands_20':band_rows,'top_residual_loci':sorted(records,key=lambda r:r['residual_energy'],reverse=True)[:100],'interpretation':'Residuals are mapped to the transcription folio/locus order. This tests sequence-layout alignment; it is not a pixel-coordinate registration because RF transcription lacks xy ink coordinates.'}
Path('/tmp/vesuvius-progress-prize/voynich_residual_layout_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps({'symbols':len(symbols),'folios':len(page_rows),'top_folios':page_rows[:15],'top_loci':res['top_residual_loci'][:10],'layout_bands':band_rows},indent=2))
