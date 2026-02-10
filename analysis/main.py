#%%

import json
import numpy as np
import matplotlib.pyplot as plt

def load(fnm):
	return json.load(open(fnm))

def get_choices(data):
	trials = data['trials']
	choices = []
	for i, trial in enumerate(trials):
		if i == 0 or trials[i]['gameIndex'] != trials[i-1]['gameIndex']:
			continue
		hole_locs = sorted(trial['holes']['hole_locations'])
		if len(hole_locs) == 2:
			h_cur = trial['holeUsed']
			h_prev = trials[i-1]['holeUsed']
			dist_L = np.abs(hole_locs[0] - h_prev)
			dist_R = np.abs(hole_locs[1] - h_prev)
			choice = hole_locs.index(h_cur)
			rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
			choices.append((dist_L, dist_R, rt, choice))
	return np.vstack(choices)

def compare_greedy_vs_rollout(data):
	trials = data['trials']
	choices = []
	for i, trial in enumerate(trials):
		if i == 0 or trials[i]['gameIndex'] != trials[i-1]['gameIndex']:
			continue
		# if trial['holes']['plan_depth'] == 2 and trial['holes']['layer_index'] == 1:
		if len(trial['holes']['hole_locations']) == 2 and len(trials[i+1]['holes']['hole_locations']) == 1 and len(trials[i-1]['holes']['hole_locations']) == 1:
			h_prev = trials[i-1]['holeUsed']
			h_cur = trial['holeUsed']
			hole_locs = sorted(trial['holes']['hole_locations'])
			h_next = trials[i+1]['holeUsed']

			dist_L1 = np.abs(hole_locs[0] - h_prev)
			dist_R1 = np.abs(hole_locs[1] - h_prev)
			dist_L2 = dist_L1 + np.abs(hole_locs[0] - h_next)
			dist_R2 = dist_R1 + np.abs(hole_locs[1] - h_next)
			
			choice = hole_locs.index(h_cur)
			rt = trial['timePassedThru'] - trials[i-1]['timePassedThru']
			choices.append((dist_L1, dist_R1, dist_L2, dist_R2, rt, choice))
	return np.vstack(choices)

def plot_psychometric_curve(X, y, fig=None, color='k', xlabel='Δ Distance to hole (L - R)', label='_'):
	xs = np.unique(X)
	ys = []
	ses = []
	for x in xs:
		ix = X == x
		ymu = np.nanmean(y[ix])
		ys.append(ymu)
		ses.append(np.nanstd(y[ix]) / np.sqrt(sum(ix)))

	if fig is None:
		plt.figure(figsize=(3,3), dpi=300)
	for x,y,se in zip(xs,ys,ses):
		plt.plot([x,x],[y-se,y+se],'-', color=color, alpha=0.3)
	plt.plot(xs, ys, '.-', color=color, label=label)
	plt.xlabel(xlabel)
	plt.ylabel('Prob. of choosing Right Hole')
	return fig

#%% load data

fnm = '../logs/jah_20251206_1350.json'
data = load(fnm)

#%% plot psychometric curve (greedy)

# plot psychometric curve
choices = get_choices(data)
X = choices[:,0] - choices[:,1]
y = choices[:,-1]
plot_psychometric_curve(X, y)

#%% plot 2D choice heatmap (L dist vs R dist, greedy)

# heatmap where choices[:,:2] are the coordinates and choices[:,2] is the value
plt.figure(figsize=(3,3), dpi=300)
# plt.tricontourf(choices[:,0], choices[:,1], choices[:,2], levels=10, cmap='viridis')

# xs = choices[:,0]
# ys = choices[:,1]
xs = choices[:,2]
ys = choices[:,3]
# xs = choices[:,0] - choices[:,1]
# ys = choices[:,2] - choices[:,3]
zs = choices[:,-1]

xs_all = np.unique(xs)
ys_all = np.unique(ys)
Z = np.full((len(ys_all), len(xs_all)), np.nan)

for i, xi in enumerate(xs_all):
	for j, yi in enumerate(ys_all):
		ix = (xs == xi) & (ys == yi)
		z = np.nanmean(zs[ix])
		Z[j, i] = (z - 0.5) * 2  # scale to -1 to 1
    # ix = np.where(xs_all == xi)[0][0]
    # iy = np.where(ys_all == yi)[0][0]
    # Z[iy, ix] = np.nanmean(zs[(xs == xi) & (ys == yi)])

# Step 2: plot with no interpolation
plt.imshow(
    Z,
    origin='lower',
    interpolation='nearest',  # <- NO smoothing
	cmap='RdBu',
    extent=[xs_all.min(), xs_all.max(), ys_all.min(), ys_all.max()]
)
plt.clim([-1,1])

# plt.xticks([0, np.max(choices[:,0])])
# plt.yticks([0, np.max(choices[:,1])])
plt.colorbar()
plt.xlabel('Distance to Left Hole')
plt.ylabel('Distance to Right Hole')
plt.show()

#%% compare greedy vs rollout heatmaps

choices = compare_greedy_vs_rollout(data)
fig = plt.figure(figsize=(3,3), dpi=300)

X = choices[:,0] - choices[:,1]
y = choices[:,-1]
plot_psychometric_curve(X, y, fig=fig, color='r')
X = choices[:,2] - choices[:,3]
y = choices[:,-1]
plot_psychometric_curve(X, y, fig=fig, color='b')

#%% plot RT

fig = plt.figure(figsize=(3,3), dpi=300)
X1 = choices[:,0] - choices[:,1]
X2 = choices[:,2] - choices[:,3]
X = X1 * X2 # positive if greedy and rollout agree, negative if they disagree
y = choices[:,4]
plot_psychometric_curve(X, y, fig=fig, color='r')
plt.xscale('symlog')
# plt.yscale('log')
plt.xlabel('Agreement between Greedy and Rollout (L-R)')
plt.ylabel('Reaction Time (ms)')

#%% fit logistic regression to predict choice from distances
# report score on held-out test set using 5-fold cross-validation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = choices[:,:4]
y = choices[:,-1]

clf = LogisticRegression()
scores = []
for i in range(5):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	scores.append(score)
print(f'Logistic regression accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}')

# print weights 
print('Logistic regression weights:')
print(clf.coef_)

#%% plot psychometric curves for greedy vs rollout

choices = compare_greedy_vs_rollout(data)
fig = plt.figure(figsize=(3,3), dpi=300)
clrs = ['g', 'r', 'b', 'm', 'k', 'c']
names = ['Greedy L', 'Greedy R', 'Rollout L', 'Rollout R', 'Greedy (L-R)', 'Rollout (L-R)']

y = choices[:,-1]

for d in range(len(names)):
	if d < 4:
		X = choices[:,d]
	elif d == 4:
		plt.legend(fontsize=6)
		fig = plt.figure(figsize=(3,3), dpi=300)
		X = choices[:,0] - choices[:,1]
	elif d == 5:
		X = choices[:,2] - choices[:,3]
	plot_psychometric_curve(X, y, fig=fig, color=clrs[d], xlabel='Distance to hole', label=names[d])

plt.legend(fontsize=6)

#%%

if __name__ == '__main__':
	fnm = 'data/falldown_data.json'
	data = load(fnm)
	choices = get_choices(data)
	plot(choices)
