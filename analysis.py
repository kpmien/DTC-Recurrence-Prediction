# %% [markdown]
# # Recurrence prediction in differentiated thyroid cancer — corrected analysis
# **Author:** Kounabé Paulin MIEN — Erciyes University, Department of Computer Engineering
#
# This notebook revisits the original course-project analysis with a stricter evaluation protocol:
#
# 1. **Data audit** — duplicate records and the relationship between *Response to initial therapy* and the outcome.
# 2. **Where the reported 98.7 % comes from** — one 80/20 split, hyperparameters chosen on the test set, versus tuning inside the training data only, and versus 50 random splits.
# 3. **Repeated nested cross-validation** (5 × 5 outer folds, 5-fold inner grid search). Encoding, feature selection, scaling and tuning are all fitted inside each training fold.
# 4. **Baselines** — the *Response* variable alone, with no model.
# 5. **Ablation** — models without *Response* (a post-treatment variable), and without *Response* and *Risk*.
# 6. **Calibration and decision-curve analysis.**
# 7. **SHAP** — which variables take over once *Response* is removed.
#
# Changes from the original notebook: nominal variables are one-hot encoded (previously alphabetical integer codes);
# Chi-square selection is a tuned step inside the pipeline (k ∈ {10, 20, all}); a logistic-regression baseline is added;
# LIME is dropped (SHAP covers the interpretation).

# %%
import os, time, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, GridSearchCV, ParameterGrid)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             recall_score, f1_score, brier_score_loss, confusion_matrix)
from sklearn.base import clone
import statsmodels.api as sm
import shap

SEED = 42
FAST = os.environ.get('FAST', '0') == '1'        # quick smoke test
N_REPEATS = 1 if FAST else 5                      # outer CV repeats
N_SPLITS_RANDOM = 5 if FAST else 50               # random 80/20 splits in Part 2
N_JOBS = -1

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().parent
DATA = ROOT / 'data' / 'Thyroid_Diff.csv'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
RES = ROOT / 'results'; RES.mkdir(exist_ok=True)

# Figure style (print, light surface). Categorical slots in fixed order.
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 9, 'font.family': 'DejaVu Sans',
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'lines.linewidth': 2,
})
def savefig(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')

# %% [markdown]
# ## 1. Data audit

# %%
raw = pd.read_csv(DATA).rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'})
print('Shape:', raw.shape)
print('Outcome:', raw['Recurred'].value_counts().to_dict())

n_dup = int(raw.duplicated().sum())
dup_rows = raw[raw.duplicated(keep=False)]
print(f'\nExact duplicate rows: {n_dup} (rows involved: {len(dup_rows)})')
print('Outcome among duplicated profiles:', dup_rows['Recurred'].value_counts().to_dict())
print('Response among duplicated profiles:', dup_rows['Response'].value_counts().to_dict())

dedup = raw.drop_duplicates().reset_index(drop=True)
print(f'\nAfter removing duplicates: {dedup.shape[0]} patients',
      dedup['Recurred'].value_counts().to_dict())

resp_order = ['Excellent', 'Indeterminate', 'Biochemical Incomplete', 'Structural Incomplete']
ct_resp = pd.crosstab(raw['Response'], raw['Recurred']).reindex(resp_order)
ct_resp['Recurrence rate'] = (ct_resp['Yes'] / ct_resp.sum(axis=1)).round(3)
print('\nResponse vs recurrence (all 383 rows):'); print(ct_resp)
ct_resp.to_csv(RES / 'response_crosstab.csv')

audit = {'n_raw': len(raw), 'n_dup': n_dup, 'n_dedup': len(dedup),
         'dup_outcome': dup_rows['Recurred'].value_counts().to_dict(),
         'yes_raw': int((raw.Recurred == 'Yes').sum()), 'yes_dedup': int((dedup.Recurred == 'Yes').sum())}

# %% [markdown]
# ## 2. Encoding and pipeline

# %%
ORDINAL = {
    'Risk': ['Low', 'Intermediate', 'High'],
    'T': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'N': ['N0', 'N1a', 'N1b'],
    'M': ['M0', 'M1'],
    'Stage': ['I', 'II', 'III', 'IVA', 'IVB'],
    'Response': resp_order,
}
BINARY = {'Gender': ('F', 'M'), 'Smoking': ('No', 'Yes'), 'Hx Smoking': ('No', 'Yes'),
          'Hx Radiotherapy': ('No', 'Yes'), 'Focality': ('Uni-Focal', 'Multi-Focal')}
NOMINAL = ['Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology']

def encode(df):
    """Deterministic, row-wise encoding (no fitted statistics -> no leakage)."""
    X = df.drop(columns='Recurred').copy()
    for c, order in ORDINAL.items():
        X[c] = pd.Categorical(X[c], categories=order, ordered=True).codes
        assert (X[c] >= 0).all(), c
    for c, (neg, pos) in BINARY.items():
        assert set(X[c]) <= {neg, pos}, c
        X[c] = (X[c] == pos).astype(int)
    y = (df['Recurred'] == 'Yes').astype(int).values
    return X, y

X_raw, y_raw = encode(raw)
X_dd, y_dd = encode(dedup)

FEATURE_SETS = {
    'Full': [c for c in X_dd.columns],
    'No Response': [c for c in X_dd.columns if c != 'Response'],
    'No Response, no Risk': [c for c in X_dd.columns if c not in ('Response', 'Risk')],
}

def make_pipeline(cols, clf, scale):
    nom = [c for c in cols if c in NOMINAL]
    enc = ColumnTransformer([('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom)],
                            remainder='passthrough', verbose_feature_names_out=False)
    return Pipeline([('enc', enc), ('sel', SelectKBest(chi2)),
                     ('sc', StandardScaler() if scale else 'passthrough'), ('clf', clf)])

def n_encoded(cols, X):
    return make_pipeline(cols, LogisticRegression(), False).named_steps['enc'].fit(X[cols]).transform(X[cols]).shape[1]

def k_grid(cols, X):
    p = n_encoded(cols, X)
    return [k for k in (10, 20) if k < p] + ['all']

MODELS = {
    'LR':  (lambda: LogisticRegression(max_iter=5000), True,
            {'clf__C': [0.01, 0.1, 1, 10]}),
    'DT':  (lambda: DecisionTreeClassifier(random_state=SEED), False,
            {'clf__max_depth': [2, 3, 4, 5, None], 'clf__min_samples_leaf': [1, 5, 10]}),
    'RF':  (lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1), False,
            {'clf__max_depth': [None, 5], 'clf__min_samples_leaf': [1, 3]}),
    'SVM': (lambda: SVC(kernel='rbf', random_state=SEED), True,
            {'clf__C': [0.1, 1, 10, 50, 100], 'clf__gamma': ['scale', 0.01, 0.1]}),
    'MLP': (lambda: MLPClassifier(max_iter=2000, random_state=SEED), True,
            {'clf__hidden_layer_sizes': [(32,), (64, 32)], 'clf__alpha': [1e-2, 1.0]}),
}

def grid_for(name, cols, X):
    g = dict(MODELS[name][2])
    # RF does its own feature weighting: keep all features to limit compute
    g['sel__k'] = ['all'] if name == 'RF' else k_grid(cols, X)
    return g

def scores_of(model, X):
    if hasattr(model, 'predict_proba') and not isinstance(model[-1], SVC):
        return model.predict_proba(X)[:, 1], True
    return model.decision_function(X), False

def metrics(y, pred, score, is_prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'AUC': roc_auc_score(y, score), 'Accuracy': accuracy_score(y, pred),
            'Balanced acc.': balanced_accuracy_score(y, pred),
            'Sensitivity': tp / (tp + fn), 'Specificity': tn / (tn + fp),
            'F1': f1_score(y, pred, zero_division=0),
            'Brier': brier_score_loss(y, score) if is_prob else np.nan,
            'FN': int(fn), 'FP': int(fp)}

for name, cols in FEATURE_SETS.items():
    print(f'{name:<22} {len(cols)} variables -> {n_encoded(cols, X_dd)} encoded columns; k grid {k_grid(cols, X_dd)}')

# %% [markdown]
# ## 3. Where does 98.7 % come from?
# Same seed-42 stratified 80/20 split as the original notebook (all 383 rows, duplicates included).

# %%
cols = FEATURE_SETS['Full']
Xtr, Xte, ytr, yte = train_test_split(X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED)

# Duplicated profiles that sit on both sides of this split
tr_keys = set(map(tuple, pd.concat([Xtr, pd.Series(ytr, index=Xtr.index, name='y')], axis=1).values.tolist()))
te_full = pd.concat([Xte, pd.Series(yte, index=Xte.index, name='y')], axis=1)
n_cross = int(sum(tuple(r) in tr_keys for r in te_full.values.tolist()))
print(f'Test rows with an identical twin in the training set: {n_cross} / {len(yte)}')
audit['n_cross_split42'] = n_cross

part2 = []
for name in ['SVM', 'RF']:
    factory, scale, _ = MODELS[name]
    grid = grid_for(name, cols, X_raw)
    # (a) "test-set tuning": pick the configuration with the best TEST AUC (as in the original Sect. 4.4)
    best = None
    for params in ParameterGrid(grid):
        m = make_pipeline(cols, factory(), scale).set_params(**params).fit(Xtr[cols], ytr)
        s, isp = scores_of(m, Xte[cols]); r = metrics(yte, m.predict(Xte[cols]), s, isp)
        key = (r['AUC'], r['Sensitivity'])
        if best is None or key > best[0]:
            best = (key, params, r)
    part2.append({'Model': name, 'Protocol': 'Split 42, tuned on test set', **best[2]})
    # (b) tuning by 5-fold CV inside the training set only
    gs = GridSearchCV(make_pipeline(cols, factory(), scale), grid, scoring='roc_auc',
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=N_JOBS).fit(Xtr[cols], ytr)
    s, isp = scores_of(gs.best_estimator_, Xte[cols])
    part2.append({'Model': name, 'Protocol': 'Split 42, tuned in training folds',
                  **metrics(yte, gs.predict(Xte[cols]), s, isp)})
    print(name, '| test-tuned params:', best[1], '| CV-tuned params:', gs.best_params_)

# Response alone on the same test split (no model)
resp_te = Xte['Response'].values
part2.append({'Model': 'Response alone', 'Protocol': 'Split 42, no training',
              **metrics(yte, (resp_te == 3).astype(int), resp_te, False)})
df_part2 = pd.DataFrame(part2)
print(df_part2[['Model', 'Protocol', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'FN', 'FP']].round(4).to_string(index=False))

# %% [markdown]
# ### 3b. Fifty random 80/20 splits (tuning inside the training part each time)

# %%
rand = []
t0 = time.time()
for i in range(N_SPLITS_RANDOM):
    a, b, ya, yb = train_test_split(X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=1000 + i)
    for name in ['SVM', 'RF']:
        factory, scale, _ = MODELS[name]
        gs = GridSearchCV(make_pipeline(cols, factory(), scale), grid_for(name, cols, X_raw), scoring='roc_auc',
                          cv=StratifiedKFold(5, shuffle=True, random_state=i), n_jobs=N_JOBS).fit(a[cols], ya)
        s, isp = scores_of(gs.best_estimator_, b[cols])
        rand.append({'split': i, 'Model': name, **metrics(yb, gs.predict(b[cols]), s, isp)})
df_rand = pd.DataFrame(rand)
df_rand.to_csv(RES / 'random_splits.csv', index=False)
print(f'{N_SPLITS_RANDOM} splits in {time.time()-t0:.0f}s')
summ_rand = df_rand.groupby('Model')[['Accuracy', 'AUC', 'Sensitivity']].agg(
    ['mean', 'std', 'min', 'max', lambda s: (s >= 0.987).mean()])
print(summ_rand.round(4))
for name in ['SVM', 'RF']:
    acc = df_rand.loc[df_rand.Model == name, 'Accuracy']
    part2.append({'Model': name, 'Protocol': f'{N_SPLITS_RANDOM} random splits (mean)',
                  **df_rand.loc[df_rand.Model == name].drop(columns=['split', 'Model']).mean().to_dict()})
df_part2 = pd.DataFrame(part2); df_part2.to_csv(RES / 'part2_protocols.csv', index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, metric, ref in zip(axes, ['Accuracy', 'AUC'], [0.987, 0.9967]):
    bins = np.linspace(0.80, 1.0, 21)
    for name, col in [('SVM', C1), ('RF', C2)]:
        ax.hist(df_rand.loc[df_rand.Model == name, metric], bins=bins, color=col, alpha=0.55,
                label=name, edgecolor='white', linewidth=1)
    ax.axvline(ref, color=INK, lw=1.2, ls='--')
    ax.annotate('originally reported (SVM)', xy=(ref, 1.0), xycoords=('data', 'axes fraction'),
                xytext=(-4, -2), textcoords='offset points', ha='right', va='top', fontsize=7.5, color=INK2)
    ax.set_xlabel(f'Test-set {metric.lower() if metric=="Accuracy" else metric}')
axes[0].set_ylabel('Number of splits'); axes[0].legend(loc='upper left')
fig.tight_layout(); savefig(fig, 'fig_random_splits'); plt.show()

# %% [markdown]
# ## 4. Repeated nested cross-validation
# Outer: 5 × repeated stratified 5-fold. Inner: 5-fold grid search on AUC. Everything (encoding, Chi-square selection,
# scaling, hyperparameters) is fitted on the outer training fold only.

# %%
def nested_cv(X, y, cols, models, n_repeats=N_REPEATS, tag=''):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows, oof = [], []
    for f, (tr, te) in enumerate(outer.split(X, y)):
        rep = f // 5
        Xa, Xb, ya, yb = X.iloc[tr][cols], X.iloc[te][cols], y[tr], y[te]
        if 'Response' in cols:   # baseline: Response alone, no training
            r = Xb['Response'].values
            rows.append({'fold': f, 'Model': 'Response alone', **metrics(yb, (r == 3).astype(int), r, False)})
        for name in models:
            factory, scale, _ = MODELS[name]
            gs = GridSearchCV(make_pipeline(cols, factory(), scale), grid_for(name, cols, X), scoring='roc_auc',
                              cv=StratifiedKFold(5, shuffle=True, random_state=f), n_jobs=N_JOBS).fit(Xa, ya)
            s, isp = scores_of(gs.best_estimator_, Xb)
            rows.append({'fold': f, 'Model': name, **metrics(yb, gs.predict(Xb), s, isp),
                         'params': json.dumps({k: str(v) for k, v in gs.best_params_.items()})})
            if isp:
                oof += [{'fold': f, 'rep': rep, 'Model': name, 'y': yy, 'p': pp} for yy, pp in zip(yb, s)]
    out = pd.DataFrame(rows); out['Features'] = tag
    return out, pd.DataFrame(oof).assign(Features=tag)

ALL_MODELS = ['LR', 'RF'] if FAST else ['LR', 'DT', 'RF', 'SVM', 'MLP']
runs, oofs = [], []
t0 = time.time()
for tag, cols in FEATURE_SETS.items():
    r, o = nested_cv(X_dd, y_dd, cols, ALL_MODELS, tag=tag)
    runs.append(r.assign(Data='Deduplicated')); oofs.append(o.assign(Data='Deduplicated'))
    print(f'{tag}: done ({time.time()-t0:.0f}s)')
# Sensitivity analysis: full feature set on all 383 rows (duplicates kept)
r, o = nested_cv(X_raw, y_raw, FEATURE_SETS['Full'], [m for m in ALL_MODELS if m in ('LR', 'RF', 'SVM')], tag='Full')
runs.append(r.assign(Data='All 383 rows')); oofs.append(o.assign(Data='All 383 rows'))
print(f'Sensitivity run done ({time.time()-t0:.0f}s)')

cv = pd.concat(runs, ignore_index=True); cv.to_csv(RES / 'nested_cv_folds.csv', index=False)
oof = pd.concat(oofs, ignore_index=True); oof.to_csv(RES / 'nested_cv_oof.csv', index=False)

# %%
MET = ['AUC', 'Balanced acc.', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Brier']
summary = (cv.groupby(['Data', 'Features', 'Model'], sort=False)[MET]
             .agg(['mean', 'std']))
summary.to_csv(RES / 'nested_cv_summary.csv')
pd.set_option('display.width', 200)
for (d, fs), g in cv.groupby(['Data', 'Features'], sort=False):
    print(f'\n=== {d} | {fs} ===')
    t = g.groupby('Model', sort=False)[MET].agg(lambda s: f'{s.mean():.3f} ± {s.std():.3f}')
    print(t.to_string())

# %% [markdown]
# ### Corrected resampled t-test (Nadeau & Bengio, 2003)

# %%
def corrected_ttest(a, b, n_train_over_test=4.0):
    d = np.asarray(a) - np.asarray(b); k = len(d)
    var = d.var(ddof=1) * (1 / k + 1 / n_train_over_test)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return d.mean(), t, p

dd = cv[cv.Data == 'Deduplicated']
tests = []
full = dd[dd.Features == 'Full']
base = full[full.Model == 'Response alone'].sort_values('fold')['AUC'].values
for m in ALL_MODELS:
    a = full[full.Model == m].sort_values('fold')['AUC'].values
    tests.append(('Full', f'{m} vs Response alone', *corrected_ttest(a, base)))
for m in ALL_MODELS:
    a = full[full.Model == m].sort_values('fold')['AUC'].values
    b = dd[(dd.Features == 'No Response') & (dd.Model == m)].sort_values('fold')['AUC'].values
    tests.append(('Full vs No Response', m, *corrected_ttest(a, b)))
best_full = full.groupby('Model')['AUC'].mean().drop('Response alone').idxmax()
for m in ALL_MODELS:
    if m == best_full: continue
    a = full[full.Model == best_full].sort_values('fold')['AUC'].values
    b = full[full.Model == m].sort_values('fold')['AUC'].values
    tests.append(('Full', f'{best_full} vs {m}', *corrected_ttest(a, b)))
df_tests = pd.DataFrame(tests, columns=['Setting', 'Comparison', 'Mean ΔAUC', 't', 'p'])
df_tests.to_csv(RES / 'corrected_ttests.csv', index=False)
print(df_tests.round(4).to_string(index=False))

# %%
order = ['Response alone'] + ALL_MODELS
fig, ax = plt.subplots(figsize=(7.2, 3.0))
fsets = list(FEATURE_SETS); cols_ = [C1, C2, C3]; w = 0.26
for j, (fs, col) in enumerate(zip(fsets, cols_)):
    data, pos = [], []
    for i, m in enumerate(order):
        v = dd[(dd.Features == fs) & (dd.Model == m)]['AUC'].values
        if len(v): data.append(v); pos.append(i + (j - 1) * w)
    bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8), flierprops=dict(marker='o', ms=3, mfc=col, mec='white'))
    for b in bp['boxes']: b.set(facecolor=col, edgecolor='white', alpha=0.85)
    ax.plot([], [], 's', color=col, ms=7, label=fs)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.set_ylabel('AUC (outer test folds)'); ax.set_ylim(0.70, 1.005)
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
fig.tight_layout(); savefig(fig, 'fig_nested_auc'); plt.show()

# %% [markdown]
# ## 5. Calibration and decision-curve analysis (out-of-fold predictions, deduplicated data)

# %%
def calib_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return slope, citl

def net_benefit(y, p, thr):
    n = len(y); out = []
    for t in thr:
        pred = p >= t; tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)

oo = oof[(oof.Data == 'Deduplicated') & (oof.rep == 0)]
cal_rows = []
for (fs, m), g in oo.groupby(['Features', 'Model'], sort=False):
    s, c = calib_stats(g.y.values, g.p.values)
    cal_rows.append({'Features': fs, 'Model': m, 'Brier': brier_score_loss(g.y, g.p),
                     'Calibration slope': s, 'Calibration-in-the-large': c})
df_cal = pd.DataFrame(cal_rows); df_cal.to_csv(RES / 'calibration.csv', index=False)
print(df_cal.round(3).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axes[0]
ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=':')
for fs, col in zip(['Full', 'No Response'], [C1, C2]):
    g = oo[(oo.Features == fs) & (oo.Model == 'LR')]
    q = pd.qcut(g.p, 8, duplicates='drop')
    agg = g.groupby(q, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'))
    ax.plot(agg.p, agg.y, 'o-', color=col, ms=5, mec='white', label=f'LR, {fs.lower()}')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed recurrence rate')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc='upper left', fontsize=8); ax.set_title('Calibration', fontsize=9)

ax = axes[1]
thr = np.linspace(0.05, 0.6, 56)
yy = oo[(oo.Features == 'Full') & (oo.Model == 'LR')].y.values
prev = yy.mean()
ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), color=INK2, lw=1, ls='--', label='Treat all')
ax.axhline(0, color=INK, lw=1, label='Treat none')
for fs, col in zip(['Full', 'No Response'], [C1, C2]):
    g = oo[(oo.Features == fs) & (oo.Model == 'LR')]
    ax.plot(thr, net_benefit(g.y.values, g.p.values, thr), color=col, label=f'LR, {fs.lower()}')
ax.set_ylim(-0.05, prev + 0.03); ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.legend(fontsize=8, loc='lower left'); ax.set_title('Decision curve', fontsize=9)
fig.tight_layout(); savefig(fig, 'fig_calibration_dca'); plt.show()

# %% [markdown]
# ## 6. SHAP — Random Forest fitted on all deduplicated patients (hyperparameters by 5-fold CV)

# %%
def group_name(col):
    for n in NOMINAL:
        if col.startswith(n + '_'): return n
    return col

shap_tables = {}
for fs in ['Full', 'No Response']:
    cols = FEATURE_SETS[fs]
    factory, scale, _ = MODELS['RF']
    gs = GridSearchCV(make_pipeline(cols, factory(), scale), grid_for('RF', cols, X_dd), scoring='roc_auc',
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=N_JOBS).fit(X_dd[cols], y_dd)
    pipe = gs.best_estimator_
    Xt = pipe[:-1].transform(X_dd[cols])
    names = pipe.named_steps['enc'].get_feature_names_out()[pipe.named_steps['sel'].get_support()]
    sv = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names)
    imp = imp.groupby([group_name(c) for c in imp.index]).sum().sort_values(ascending=False)
    shap_tables[fs] = imp
    print(f'\n{fs} (params {gs.best_params_}):'); print(imp.round(4).head(10).to_string())
df_shap = pd.DataFrame(shap_tables).fillna(0); df_shap.to_csv(RES / 'shap_importance.csv')

# %%
top = df_shap.max(axis=1).sort_values(ascending=False).head(10).index[::-1]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
yp = np.arange(len(top)); h = 0.38
ax.barh(yp + h / 2, df_shap.loc[top, 'Full'], h, color=C1, label='Full', edgecolor='white')
ax.barh(yp - h / 2, df_shap.loc[top, 'No Response'], h, color=C2, label='No Response', edgecolor='white')
ax.set_yticks(yp); ax.set_yticklabels(top); ax.grid(axis='y', visible=False)
ax.set_xlabel('Mean |SHAP value| (Random Forest)'); ax.legend(loc='lower right')
fig.tight_layout(); savefig(fig, 'fig_shap'); plt.show()

# %% [markdown]
# ## 7. Key numbers for the manuscript

# %%
def ms(fs, m, met='AUC', data='Deduplicated'):
    v = cv[(cv.Data == data) & (cv.Features == fs) & (cv.Model == m)][met]
    return round(v.mean(), 3), round(v.std(), 3)

key = {'audit': audit,
       'rule_full_data': metrics(y_raw, (X_raw.Response.values == 3).astype(int), X_raw.Response.values, False),
       'rule_dedup': metrics(y_dd, (X_dd.Response.values == 3).astype(int), X_dd.Response.values, False),
       'best_full_model': best_full,
       'auc': {fs: {m: ms(fs, m) for m in (['Response alone'] if fs == 'Full' else []) + ALL_MODELS}
               for fs in FEATURE_SETS}}
json.dump(key, open(RES / 'key_numbers.json', 'w'), indent=2, default=float)
print(json.dumps(key, indent=2, default=float))
