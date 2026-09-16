#!/usr/bin/env python3
import json,math,re
from pathlib import Path
import numpy as np
PHI=(1+math.sqrt(5))/2; PHI2=PHI+1
src=Path('/tmp/vesuvius-progress-prize/voynich_rf1b-e.txt')
# Full RF symbol stream: alphabetic transliteration symbols, preserving order.
stream=[]
for raw in src.read_text(errors='replace').splitlines():
 m=re.match(r'<([^>]+)>\s+(.*)',raw)
 if m and '.' in m.group(1): stream.extend(re.findall(r'[A-Za-z]',m.group(2)))
N=len(stream); alphabet=sorted(set(stream)); A=np.zeros((len(alphabet),N),dtype=np.float64)
for i,ch in enumerate(stream): A[alphabet.index(ch),i]=1.0
# Remove channel DC so the spectrum measures transition structure, not frequency alone.
X=np.fft.rfft(A-A.mean(axis=1,keepdims=True),axis=1); power=np.sum(np.abs(X)**2,axis=0); power[0]=0
# Phi-transition operator: T x[n] = phi^-1 x[n-1] + phi^-2 x[n+1].
# Fourier eigenvalue lambda(k)=phi^-1 exp(-i k)+phi^-2 exp(i k).
k=2*np.pi*np.arange(len(power))/N
lam=(PHI**-1)*np.exp(-1j*k)+(PHI**-2)*np.exp(1j*k)
# Scale-invariant spectral coordinate: normalize eigenvalue magnitude by lambda(0).
lam0=abs(PHI**-1+PHI**-2); invariant=np.abs(lam)/lam0
# Aggregate power into phi-rungs of inverse wavelength: rung(log frequency / log phi).
freq=np.arange(len(power))/N; nonzero=freq>0
freq_rung=np.log(np.maximum(freq,1/N))/math.log(PHI)
bins={}
for p,r in zip(power[nonzero],freq_rung[nonzero]): bins[str(int(round(r)))] = bins.get(str(int(round(r))),0.0)+float(p)
# top spectral modes excluding DC
idx=np.argsort(power[1:])[-20:][::-1]+1
top=[{'bin':int(i),'normalized_frequency':float(freq[i]),'frequency_rung':float(freq_rung[i]),'operator_eigenvalue_real':float(lam[i].real),'operator_eigenvalue_imag':float(lam[i].imag),'operator_eigenvalue_abs_normalized':float(invariant[i]),'power_fraction':float(power[i]/power.sum())} for i in idx]
# Per-symbol dominant spectral power and phase coherence under the same operator.
per=[]
for j,ch in enumerate(alphabet):
 q=np.abs(X[j])**2; q[0]=0; ii=int(np.argmax(q)); per.append({'symbol':ch,'count':int(A[j].sum()),'dominant_frequency':float(freq[ii]),'dominant_frequency_rung':float(freq_rung[ii]),'power_fraction':float(q.sum()/power.sum()),'phase':float(np.angle(X[j,ii]))})
res={'source':str(src),'symbols':N,'alphabet':alphabet,'phi':PHI,'phi_squared':PHI2,'identity_residual':PHI2-(PHI+1),'operator':'T x[n] = phi^-1 x[n-1] + phi^-2 x[n+1]','operator_scale_reference':lam0,'operator_eigenvalue':'lambda(k)=phi^-1 exp(-ik)+phi^-2 exp(ik)','scale_invariant_coordinate':'abs(lambda(k))/abs(lambda(0))','total_non_dc_power':float(power.sum()),'spectral_rung_mass':bins,'top_modes':top,'per_symbol_modes':per,'interpretation':'The decomposition is invariant under global phi rescaling of the operator weights; frequency positions are expressed as phi-rungs. It describes transition structure in the transcription stream and is not a plaintext translation.'}
Path('/tmp/vesuvius-progress-prize/voynich_phi_spectral_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps({'symbols':N,'alphabet':alphabet,'operator':res['operator'],'top_modes':top[:10],'spectral_rung_mass':bins},indent=2))
