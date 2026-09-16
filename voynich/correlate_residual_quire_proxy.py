#!/usr/bin/env python3
import json,math,re
from pathlib import Path
import numpy as np
PHI=(1+math.sqrt(5))/2
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt')
records=[]; syms=[]
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if not m or '.' not in m.group(1): continue
 locus=m.group(1); text=m.group(2); chars=re.findall(r'[A-Za-z]',re.sub(r'<->|\{[^}]*\}|\[[^]]*\]|@\d+','',text))
 if not chars: continue
 records.append((locus,locus.split('.')[0],len(syms),len(chars))); syms.extend(chars)
N=len(syms); X=np.zeros((len(set(syms)),N)); alpha=sorted(set(syms)); ix={c:i for i,c in enumerate(alpha)}
for n,c in enumerate(syms): X[ix[c],n]=1
pred=PHI**-1*np.roll(X,1,axis=1)+PHI**-2*np.roll(X,-1,axis=1); E=np.sum((X-pred)**2,axis=0)
bound=np.zeros(N); folios=[]
for locus,page,start,n in records:
 if not folios or page!=folios[-1][0]: folios.append((page,start)); bound[start]=1
# correlation via centered convolution; report lags in symbols.
a=E-E.mean(); b=bound-bound.mean(); corr=np.correlate(a,b,mode='full'); lags=np.arange(-N+1,N); corr_norm=corr/(np.sqrt(np.sum(a*a)*np.sum(b*b))+1e-12); keep=np.abs(lags)<=2000; ii=np.where(keep)[0]; top=ii[np.argsort(np.abs(corr_norm[ii]))[-20:][::-1]]
# boundary-neighborhood vs interior, with 100-symbol window.
w=100; near=np.zeros(N,bool)
for p,_ in folios:
 start=next(s for pg,s in folios if pg==p); near[max(0,start-w):min(N,start+w+1)]=1
near_mean=float(E[near].mean()); interior_mean=float(E[~near].mean());
rng=np.random.default_rng(42); diffs=[]
for _ in range(2000): diffs.append(float(E[rng.permutation(near)].mean()-interior_mean))
obs=near_mean-interior_mean
res={'symbols':N,'folios':len(folios),'boundary_definition':'first transcription record of each folio; quire/bifolio metadata unavailable in RF file','boundary_count':len(folios),'residual_operator':'epsilon=x-(phi^-1 previous + phi^-2 next)','top_cross_correlation_lags':[{'lag_symbols':int(lags[i]),'correlation':float(corr_norm[i])} for i in top],'zero_lag_correlation':float(corr_norm[lags==0][0]),'boundary_neighborhood_window_symbols':w,'near_boundary_mean_residual_energy':near_mean,'interior_mean_residual_energy':interior_mean,'difference':obs,'permutation_p_two_sided':float(np.mean(np.abs(diffs)>=abs(obs))),'conclusion':'folio-boundary proxy correlation computed; it is not a quire-boundary test without independent quire metadata'}
Path('/tmp/vesuvius-progress-prize/voynich_residual_quire_proxy_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
