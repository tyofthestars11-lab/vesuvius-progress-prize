#!/usr/bin/env python3
import json,math,re
from pathlib import Path
import numpy as np
PHI=(1+math.sqrt(5))/2
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt'); records=[]; syms=[]
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if not m or '.' not in m.group(1): continue
 loc=m.group(1); text=m.group(2); chars=re.findall(r'[A-Za-z]',re.sub(r'<->|\{[^}]*\}|\[[^]]*\]|@\d+','',text))
 if not chars: continue
 fm=re.match(r'f(\d+)',loc); folio=int(fm.group(1)) if fm else -1
 records.append((loc,folio,len(syms),len(chars))); syms.extend(chars)
N=len(syms); alpha=sorted(set(syms)); X=np.zeros((len(alpha),N)); ix={c:i for i,c in enumerate(alpha)}
for n,c in enumerate(syms): X[ix[c],n]=1
E=np.sum((X-(PHI**-1*np.roll(X,1,axis=1)+PHI**-2*np.roll(X,-1,axis=1)))**2,axis=0)
# documented folio omissions from the physical description's 1..116 foliation sequence.
missing={12,59,60,61,62,63,64,74,91,92,97,98,109,110}
# Assign first symbol position of each extant folio.
folio_start={};
for loc,f,s,n in records: folio_start.setdefault(f,s)
quire_bound=np.zeros(N); bifolio_bound=np.zeros(N)
for f,s in folio_start.items():
 # standard quire boundaries at original foliation starts 1,9,17,...; boundary at every 4-folio bifolio pair.
 if f>=1 and (f-1)%8==0: quire_bound[s]=1
 if f>=1 and (f-1)%2==0: bifolio_bound[s]=1

def analyze(bound,name):
 a=E-E.mean(); b=bound-bound.mean(); c=np.correlate(a,b,mode='full'); l=np.arange(-N+1,N); cn=c/(math.sqrt(np.sum(a*a)*np.sum(b*b))+1e-12); mask=np.abs(l)<=5000; ii=np.where(mask)[0]; top=ii[np.argsort(np.abs(cn[ii]))[-10:][::-1]]
 w=100; near=np.zeros(N,bool); starts=np.where(bound>0)[0]
 for s in starts: near[max(0,s-w):min(N,s+w+1)]=1
 nm=float(E[near].mean()); im=float(E[~near].mean()); obs=nm-im; rng=np.random.default_rng(42); null=[]
 for _ in range(2000):
  perm=rng.permutation(near); null.append(float(E[perm].mean()-im))
 return {'boundary_count':int(bound.sum()),'zero_lag_correlation':float(cn[l==0][0]),'top_lags':[{'lag_symbols':int(l[i]),'correlation':float(cn[i])} for i in top],'window':w,'near_mean':nm,'interior_mean':im,'difference':obs,'permutation_p_two_sided':float(np.mean(np.abs(null)>=abs(obs)))}
res={'source':str(src),'documented_codicology':'standard quire = four bifolios = eight folios; bifolio = two folios; 102 extant of originally probably 116; 18 extant of 20 quires; foldouts present; missing folios listed in model','missing_original_folios':sorted(missing),'symbols':N,'folio_count':len(folio_start),'quire_model':'original foliation starts 1,9,17,...,113','bifolio_model':'original foliation starts 1,3,5,...,115','quire_boundaries':analyze(quire_bound,'quire model'),'bifolio_boundaries':analyze(bifolio_bound,'bifolio model'),'status':'model-based physical gathering analysis; exact historical quire assignments and foldout panel topology are not encoded in RF transcription'}
Path('/tmp/vesuvius-progress-prize/voynich_true_codicology_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
