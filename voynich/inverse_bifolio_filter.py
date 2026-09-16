#!/usr/bin/env python3
import json,math,re
from pathlib import Path
import numpy as np
PHI=(1+math.sqrt(5))/2
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt'); records=[]; syms=[]
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if not m or '.' not in m.group(1): continue
 loc=m.group(1); chars=re.findall(r'[A-Za-z]',re.sub(r'<->|\{[^}]*\}|\[[^]]*\]|@\d+','',m.group(2)))
 if chars:
  fm=re.match(r'f(\d+)',loc); fol=int(fm.group(1)) if fm else -1
  records.append((loc,fol,len(syms),len(chars))); syms.extend(chars)
N=len(syms); alpha=sorted(set(syms)); X=np.zeros((len(alpha),N)); ix={c:i for i,c in enumerate(alpha)}
for n,c in enumerate(syms): X[ix[c],n]=1
E=np.sum((X-(PHI**-1*np.roll(X,1,axis=1)+PHI**-2*np.roll(X,-1,axis=1)))**2,axis=0); z=E-E.mean(); F=np.fft.rfft(z); freq=np.fft.rfftfreq(N)
# Primary wavelet bifolio scales and a secondary high-scale harmonic.
scales=[90.50966799187809,107.63474115247546,1722.1558584396073]
bands=[]; components=[]
for s in scales:
 center=6/(2*math.pi*s); width=center*0.16; mask=(freq>=center-width)&(freq<=center+width); G=np.zeros_like(F); G[mask]=F[mask]; comp=np.fft.irfft(G,n=N); components.append(comp); bands.append({'scale':s,'scale_rung':math.log(s)/math.log(PHI),'center_frequency':center,'bandwidth':width,'bins':int(mask.sum()),'variance':float(comp.var())})
# Primary pure component = sum of the two nearby dominant bands; secondary kept separate.
pure=components[0]+components[1]; secondary=components[2];
folio_starts={}
for loc,fol,start,n in records: folio_starts.setdefault(fol,start)
bif=np.zeros(N)
for fol,start in folio_starts.items():
 if fol>=1 and (fol-1)%2==0: bif[start]=1
# Correlate filtered component with boundary train and compare against raw residual.
def corr(a,b): return float(np.corrcoef(a,b)[0,1])
def neighborhood(sig,w=100):
 near=np.zeros(N,bool)
 for p in np.where(bif>0)[0]: near[max(0,p-w):min(N,p+w+1)]=1
 return float(sig[near].mean()),float(sig[~near].mean())
res={'source':str(src),'operator':'epsilon=x-(phi^-1 previous + phi^-2 next)','phi':PHI,'input_symbols':N,'filter':'inverse real FFT after symmetric positive-frequency band selection','primary_bifolio_bands':bands[:2],'secondary_harmonic_band':bands[2],'primary_component_variance':float(pure.var()),'secondary_component_variance':float(secondary.var()),'raw_residual_variance':float(z.var()),'primary_energy_fraction':float(np.sum(pure*pure)/np.sum(z*z)),'secondary_energy_fraction':float(np.sum(secondary*secondary)/np.sum(z*z)),'primary_boundary_correlation':corr(pure,bif),'secondary_boundary_correlation':corr(secondary,bif),'raw_boundary_correlation':corr(z,bif),'primary_near_boundary_mean':neighborhood(pure)[0],'primary_interior_mean':neighborhood(pure)[1],'signal_file':'voynich_pure_bifolio_component.npy','secondary_file':'voynich_secondary_harmonic_component.npy','interpretation':'The primary inverse-Fourier component is the measured bifolio-band signal selected from the wavelet peaks; pure means band-limited, not proof that all surviving power is physical bifolio structure.'}
np.save('/tmp/vesuvius-progress-prize/voynich_pure_bifolio_component.npy',pure.astype(np.float32)); np.save('/tmp/vesuvius-progress-prize/voynich_secondary_harmonic_component.npy',secondary.astype(np.float32)); Path('/tmp/vesuvius-progress-prize/voynich_inverse_bifolio_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
