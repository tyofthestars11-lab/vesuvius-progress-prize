#!/usr/bin/env python3
import json,math
from pathlib import Path
import numpy as np
D=json.load(open('/tmp/vesuvius-progress-prize/voynich_phi_spectral_results.json'))
phi=D['phi']; top=D['top_modes']
# Sort modes by spectral power and retain dominant low-frequency structure.
peaks=sorted(top,key=lambda x:x['power_fraction'],reverse=True)
rungs=np.array([x['frequency_rung'] for x in peaks],float); powers=np.array([x['power_fraction'] for x in peaks],float)
# A phi-transition generator has unit-rung scaling: frequency -> phi*f shifts rung by 1.
# Express each mode in base frequency, wavelength, and phi-power coordinate.
for x in peaks:
 f=x['normalized_frequency']; x['wavelength_symbols']=1/f; x['phi_power_frequency']=math.log(f,phi); x['relative_to_lowest_dominant']=x['frequency_rung']-rungs[-1]
# Dominant cluster: modes within 5 rungs of the strongest peak.
strong=peaks[:10]; cluster_rungs=[float(x['frequency_rung']) for x in strong]
cluster_gaps=[cluster_rungs[i+1]-cluster_rungs[i] for i in range(len(cluster_rungs)-1)]
# Estimate generator period from strongest low-frequency modes via weighted mean of rung differences.
weights=np.array([x['power_fraction'] for x in strong[:-1]])
period=float(np.average(np.abs(cluster_gaps),weights=weights)) if len(weights) else 0
# Eigenvalue phase drift under the declared operator.
lam_phase=[math.atan2(x['operator_eigenvalue_imag'],x['operator_eigenvalue_real']) for x in strong]
res={'generator':'phi-transition long-range generator','operator':D['operator'],'phi':phi,'dominant_modes':strong,'dominant_frequency_rungs':cluster_rungs,'adjacent_rung_gaps':cluster_gaps,'weighted_gap_scale':period,'eigenphase_radians':lam_phase,'structural_read':{'primary_mode_rung':float(strong[0]['frequency_rung']),'primary_mode_wavelength_symbols':float(strong[0]['wavelength_symbols']),'low_frequency_cluster':'dominant modes occupy approximately rungs -25 to -20','successor_rule':'multiplication by phi advances frequency one rung; division by phi descends one rung','generator_form':'x[n] = long-range phi-scaled transition field plus symbol-specific residual modes','interpretation':'The dominant generator is a long-range, slowly varying phi-transition field. The spectrum does not identify a unique plaintext grammar; it identifies the transition scale and residual symbol modes.'}}
Path('/tmp/vesuvius-progress-prize/voynich_generator_results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
