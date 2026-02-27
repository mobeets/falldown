#%%

import numpy as np
import json
import matplotlib.pyplot as plt

#%% load plans

fnm = '/Users/mobeets/Downloads/plans (6).json'
data = json.load(open(fnm, 'r'))
plans2 = np.array(data['plans'][0])
plans3 = np.array(data['plans'][1])

print(len(plans2), len(plans3))

#%% summarize plan counts

plans = plans3

plan_summary = {}
pts = []
for plan in plans:
    d1 = plan['oneStepPathL'] - plan['oneStepPathR']
    d2 = plan['multiStepPathL'] - plan['multiStepPathR']
    d_key = (d1, d2)
    pts.append((d1,d2, plan['oneStepPathL'], plan['oneStepPathR'], plan['multiStepPathL'], plan['multiStepPathR']))

    if plan['oneStepChoice'] == plan['multiStepChoice']:
        key = 'same - L/M/R'
    elif plan['oneStepChoice'] != plan['multiStepChoice']:
        key = 'different L/M/R'
    
    if key not in plan_summary:
        plan_summary[key] = 0
    plan_summary[key] += 1
pts = np.vstack(pts)

for k, v in plan_summary.items():
    print(f"{k}: {v}")

#%%

# xs = pts[:,0]; ys = pts[:,1]; xlbl = 'One-step L-R distance'; ylbl = 'Multi-step L-R distance'
xs = pts[:,2]; ys = pts[:,3]; xlbl = 'One-step L distance'; ylbl = 'One-step R distance'
# xs = pts[:,4]; ys = pts[:,5]; xlbl = 'Multi-step L distance'; ylbl = 'Multi-step R distance'

xs_all = np.unique(xs)
ys_all = np.unique(ys)
Z = np.full((len(ys_all), len(xs_all)), np.nan)

for i, xi in enumerate(xs_all):
    for j, yi in enumerate(ys_all):
        ix = (xs == xi) & (ys == yi)
        z = sum(ix)
        Z[j, i] = z

plt.figure(figsize=(2,3), dpi=300)
plt.imshow(
    Z,
    origin='lower',
    interpolation='nearest',  # <- NO smoothing
	cmap='Reds',
    extent=[xs_all.min(), xs_all.max(), ys_all.min(), ys_all.max()]
)
plt.xlabel(xlbl)
plt.ylabel(ylbl)
plt.colorbar()
