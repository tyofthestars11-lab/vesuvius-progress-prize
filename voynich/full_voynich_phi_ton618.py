#!/usr/bin/env python3
import json,math,re
from pathlib import Path
PHI=(1+math.sqrt(5))/2; PHI2=PHI+1
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt')
lines=[]; total=0
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if not m or '.' not in m.group(1): continue
 locus=m.group(1); text=m.group(2)
 # RF/EVA payload: count alphabetic transliteration symbols; preserve dots/braces as separators.
 payload=re.sub(r'<->|\{[^}]*\}|\[[^]]*\]|@\d+','',text)
 count=len(re.findall(r'[A-Za-z]',payload));
 if count<=0: continue
 total+=count; lines.append({'locus':locus,'symbol_count':count,'cumulative_symbols':total,'rung':math.log(total)/math.log(PHI)})
ton_mass=4.07e10; ton_rung=math.log(ton_mass)/math.log(PHI)
for x in lines: x['relative_to_TON618']=x['rung']-ton_rung
# page/fold aggregation
pages={}
for x in lines:
 page=x['locus'].split('.')[0]; pages.setdefault(page,0); pages[page]+=x['symbol_count']
pager=[]; c=0
for p,n in pages.items():
 c+=n; pager.append({'page':p,'symbols':n,'cumulative':c,'rung':math.log(c)/math.log(PHI),'relative_to_TON618':math.log(c)/math.log(PHI)-ton_rung})
res={'source':str(src),'phi':PHI,'phi_squared':PHI2,'identity_residual':PHI2-(PHI+1),'TON618_mass_solar':ton_mass,'TON618_rung':ton_rung,'lines':len(lines),'pages':len(pager),'total_transcription_symbols':total,'natural_full_manuscript_rung_S0_1':math.log(total)/math.log(PHI),'relative_full_rung_to_TON618':math.log(total)/math.log(PHI)-ton_rung,'line_rung_first':lines[0] if lines else None,'line_rung_last':lines[-1] if lines else None,'page_rungs':pager,'interpretation':'The full RF transcription maps to a cumulative phi-rung information line anchored to TON618. It is a structural transform of Voynichese, not a plaintext linguistic decipherment.'}
Path('/tmp/vesuvius-progress-prize/full_voynich_phi_ton618_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps({k:res[k] for k in ['phi','identity_residual','TON618_rung','lines','pages','total_transcription_symbols','natural_full_manuscript_rung_S0_1','relative_full_rung_to_TON618']},indent=2))
