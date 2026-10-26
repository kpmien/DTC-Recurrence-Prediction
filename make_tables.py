"""Generate LaTeX tables and number macros for the manuscript from results/*.csv.
Every number quoted in the paper comes from here, so text and notebook cannot drift apart."""
import json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / 'results', ROOT / 'paper'
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
key = json.load(open(RES / 'key_numbers.json'))
cv = pd.read_csv(RES / 'nested_cv_folds.csv')
p2 = pd.read_csv(RES / 'part2_protocols.csv')
rnd = pd.read_csv(RES / 'random_splits.csv')
tt = pd.read_csv(RES / 'corrected_ttests.csv')
cal = pd.read_csv(RES / 'calibration.csv')
shp = pd.read_csv(RES / 'shap_importance.csv', index_col=0)
xt = pd.read_csv(RES / 'response_crosstab.csv', index_col=0)

macros = {}
def m(name, value): macros[name] = value
f3 = lambda x: f'{x:.3f}'
pct = lambda x: f'{100 * x:.1f}\\%'
def pval(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'

# ---- audit ---------------------------------------------------------------
a = key['audit']
m('nRaw', a['n_raw']); m('nDup', a['n_dup']); m('nDedup', a['n_dedup']); m('nCross', a['n_cross_split42'])
m('nDupRows', sum(a['dup_outcome'].values())); m('nYes', a['yes_dedup'])
m('nNoDedup', a['n_dedup'] - a['yes_dedup'])
for lvl, tag in [('Excellent', 'Exc'), ('Indeterminate', 'Ind'), ('Biochemical Incomplete', 'Bio'), ('Structural Incomplete', 'Str')]:
    r = xt.loc[lvl]
    m(f'resp{tag}Yes', int(r['Yes'])); m(f'resp{tag}N', int(r['Yes'] + r['No'])); m(f'resp{tag}Rate', pct(r['Recurrence rate']))
for k, tag in [('rule_full_data', 'Raw'), ('rule_dedup', 'Dd')]:
    r = key[k]
    m(f'rule{tag}Acc', pct(r['Accuracy'])); m(f'rule{tag}Sens', pct(r['Sensitivity']))
    m(f'rule{tag}Spec', pct(r['Specificity'])); m(f'rule{tag}AUC', f3(r['AUC']))
    m(f'rule{tag}FN', r['FN']); m(f'rule{tag}FP', r['FP'])
m('ruleRawCorrect', int(round(key['rule_full_data']['Accuracy'] * a['n_raw'])))

xt_tab = ['\\begin{table}[!htbp]', '\\centering',
          '\\caption{Recurrence (Rec.) by response to initial therapy (all 383 records).}', '\\label{tab:response}',
          '\\footnotesize', '\\begin{tabular}{@{}lrrr@{}}', '\\toprule',
          'Response to therapy & No rec. & Rec. & Rate \\\\', '\\midrule']
for lvl in xt.index:
    r = xt.loc[lvl]
    xt_tab.append(f'{lvl} & {int(r["No"])} & {int(r["Yes"])} & {pct(r["Recurrence rate"])} \\\\')
xt_tab += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'tables/response.tex').write_text('\n'.join(xt_tab) + '\n')

# ---- protocols -----------------------------------------------------------
prot_tab = ['\\begin{table*}[!htbp]', '\\centering',
            '\\caption{Test-set performance of the SVM and RF under increasingly strict protocols (all 383 records, full feature set). '
            'Rows A and B use the original seed-42 split (77 test patients). Row C is the mean over 50 random 80/20 splits. '
            'The first row gives the figures reported in the original analysis.}',
            '\\label{tab:protocols}', '\\small', '\\begin{tabularx}{\\linewidth}{@{}l X r r r r@{}}', '\\toprule',
            'Model & Protocol & Accuracy & AUC & Sensitivity & Specificity \\\\', '\\midrule',
            'SVM & Original analysis (seed-42 split, tuned on the test set) & 0.987 & 0.997 & 0.955 & 1.000 \\\\', '\\midrule']
plabel = {'Split 42, tuned on test set': '(A) Seed-42 split, tuned on the test set',
          'Split 42, tuned in training folds': '(B) Seed-42 split, tuned in training folds',
          'Split 42, no training': 'Seed-42 split, no training'}
for model in ['SVM', 'RF', 'Response alone']:
    for _, r in p2[p2.Model == model].iterrows():
        lab = plabel.get(r.Protocol, '(C) 50 random splits, tuned in training folds (mean)')
        prot_tab.append(f'{model} & {lab} & {f3(r.Accuracy)} & {f3(r.AUC)} & {f3(r.Sensitivity)} & {f3(r.Specificity)} \\\\')
    if model != 'Response alone': prot_tab.append('\\addlinespace')
prot_tab += ['\\bottomrule', '\\end{tabularx}', '\\end{table*}']
(OUT / 'tables/protocols.tex').write_text('\n'.join(prot_tab) + '\n')

for model, tag in [('SVM', 'Svm'), ('RF', 'Rf')]:
    A = p2[(p2.Model == model) & (p2.Protocol == 'Split 42, tuned on test set')].iloc[0]
    B = p2[(p2.Model == model) & (p2.Protocol == 'Split 42, tuned in training folds')].iloc[0]
    m(f'protA{tag}Acc', f3(A.Accuracy)); m(f'protA{tag}AUC', f3(A.AUC)); m(f'protA{tag}FN', int(A.FN))
    m(f'protB{tag}Acc', f3(B.Accuracy)); m(f'protB{tag}AUC', f3(B.AUC)); m(f'protB{tag}FN', int(B.FN))
    s = rnd[rnd.Model == model]
    m(f'rnd{tag}AccMean', f3(s.Accuracy.mean())); m(f'rnd{tag}AccSd', f3(s.Accuracy.std()))
    m(f'rnd{tag}AccMin', f3(s.Accuracy.min())); m(f'rnd{tag}AccMax', f3(s.Accuracy.max()))
    m(f'rnd{tag}AUCMean', f3(s.AUC.mean())); m(f'rnd{tag}AUCSd', f3(s.AUC.std()))
    m(f'rnd{tag}AUCMin', f3(s.AUC.min())); m(f'rnd{tag}AUCMax', f3(s.AUC.max()))
    m(f'rnd{tag}NgeOrig', int((s.Accuracy >= 0.987 - 1e-9).sum()))
R = p2[p2.Model == 'Response alone'].iloc[0]
m('protRespAcc', f3(R.Accuracy)); m('protRespAUC', f3(R.AUC)); m('protRespFN', int(R.FN))
m('nRandom', rnd.split.nunique())

# ---- nested CV main table -------------------------------------------------
dd = cv[cv.Data == 'Deduplicated']
order = ['Response alone', 'LR', 'DT', 'RF', 'SVM', 'MLP']
names = {'Response alone': 'Response alone$^{a}$', 'LR': 'Logistic regression', 'DT': 'Decision tree',
         'RF': 'Random forest', 'SVM': 'SVM (RBF)', 'MLP': 'Multilayer perceptron'}
def msd(s, d=3):
    return f'{s.mean():.{d}f} $\\pm$ {s.std():.{d}f}' if s.notna().any() else '--'
main = ['\\begin{table*}[!htbp]', '\\centering',
        f'\\caption{{Repeated nested cross-validation on the {a["n_dedup"]} unique records, full feature set '
        '(mean $\\pm$ SD over 25 outer test folds).}', '\\label{tab:nested}', '\\footnotesize',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{@{}lccccc@{}}', '\\toprule',
        'Model & AUC & Balanced acc. & Sensitivity & Specificity & Brier \\\\', '\\midrule']
full = dd[dd.Features == 'Full']
for mo in order:
    g = full[full.Model == mo]
    main.append(f'{names[mo]} & {msd(g.AUC)} & {msd(g["Balanced acc."])} & {msd(g.Sensitivity)} & '
                f'{msd(g.Specificity)} & {msd(g.Brier)} \\\\')
main += ['\\bottomrule', '\\end{tabular}',
         '\\par\\smallskip\\raggedright\\footnotesize $^{a}$ Ordinal response code used as the score (AUC); '
         'threshold metrics use the rule ``recurrence if and only if structural incomplete\'\'. No training. '
         'The SVM has no Brier score because it outputs decision values, not probabilities.',
         '\\end{table*}']
(OUT / 'tables/nested.tex').write_text('\n'.join(main) + '\n')

for mo in order:
    tag = {'Response alone': 'Resp'}.get(mo, mo.title() if mo not in ('LR', 'DT', 'RF', 'SVM', 'MLP') else
                                          {'LR': 'Lr', 'DT': 'Dt', 'RF': 'Rf', 'SVM': 'Svm', 'MLP': 'Mlp'}[mo])
    for fs, ftag in [('Full', 'Full'), ('No Response', 'NoResp'), ('No Response, no Risk', 'NoRisk')]:
        g = dd[(dd.Features == fs) & (dd.Model == mo)]
        if len(g) == 0: continue
        m(f'cv{tag}{ftag}AUC', f3(g.AUC.mean())); m(f'cv{tag}{ftag}AUCsd', f3(g.AUC.std()))
        m(f'cv{tag}{ftag}Sens', f3(g.Sensitivity.mean())); m(f'cv{tag}{ftag}Spec', f3(g.Specificity.mean()))
        m(f'cv{tag}{ftag}Bacc', f3(g['Balanced acc.'].mean()))
        m(f'cv{tag}{ftag}Acc', f3(g.Accuracy.mean()))
best = key['best_full_model']
m('bestModel', {'LR': 'logistic regression', 'DT': 'decision tree', 'RF': 'random forest', 'SVM': 'SVM', 'MLP': 'multilayer perceptron'}[best]); m('bestModelShort', best)

# ---- ablation table -------------------------------------------------------
abl = ['\\begin{table*}[!htbp]', '\\centering',
       '\\caption{AUC (mean $\\pm$ SD over 25 outer folds) with and without the post-treatment variable \\emph{Response}, '
       f'{a["n_dedup"]} unique records. $\\Delta$ and $p$ compare the full and the No-Response feature sets '
       '(corrected resampled $t$-test).}',
       '\\label{tab:ablation}', '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
       '\\begin{tabular}{@{}lccccc@{}}', '\\toprule',
       'Model & Full & No Response & No Response, no Risk & $\\Delta$AUC & $p$ \\\\', '\\midrule']
for mo in order[1:]:
    cells = [msd(dd[(dd.Features == fs) & (dd.Model == mo)].AUC) for fs in ['Full', 'No Response', 'No Response, no Risk']]
    r = tt[(tt.Setting == 'Full vs No Response') & (tt.Comparison == mo)].iloc[0]
    abl.append(f'{names[mo]} & {" & ".join(cells)} & {r["Mean ΔAUC"]:+.3f} & {pval(r.p)} \\\\')
abl += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
(OUT / 'tables/ablation.tex').write_text('\n'.join(abl) + '\n')

# ---- tests ----------------------------------------------------------------
tst = ['\\begin{table}[!htbp]', '\\centering',
       '\\caption{Pairwise AUC comparisons on the full feature set (corrected resampled $t$-test, 25 outer folds).}',
       '\\label{tab:tests}', '\\small', '\\begin{tabular}{@{}lrrr@{}}', '\\toprule',
       'Comparison & Mean $\\Delta$AUC & $t$ & $p$ \\\\', '\\midrule']
for _, r in tt[tt.Setting == 'Full'].iterrows():
    tst.append(f'{r.Comparison} & {r["Mean ΔAUC"]:+.3f} & {r.t:.2f} & {pval(r.p)} \\\\')
tst += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'tables/tests.tex').write_text('\n'.join(tst) + '\n')
for _, r in tt.iterrows():
    if r.Setting == 'Full' and r.Comparison.endswith('vs Response alone'):
        mo = r.Comparison.split()[0]
        m(f'tResp{ {"LR":"Lr","DT":"Dt","RF":"Rf","SVM":"Svm","MLP":"Mlp"}[mo] }D', f'{r["Mean ΔAUC"]:+.3f}')
        m(f'tResp{ {"LR":"Lr","DT":"Dt","RF":"Rf","SVM":"Svm","MLP":"Mlp"}[mo] }P', pval(r.p))
m('nSigVsResp', int(((tt.Setting == 'Full') & tt.Comparison.str.endswith('vs Response alone') & (tt.p < 0.05)).sum()))
m('maxGainVsResp', f'{tt[(tt.Setting == "Full") & tt.Comparison.str.endswith("vs Response alone")]["Mean ΔAUC"].max():.3f}')
abl_t = tt[tt.Setting == 'Full vs No Response']
m('ablMinDrop', f'{abl_t["Mean ΔAUC"].min():.3f}'); m('ablMaxDrop', f'{abl_t["Mean ΔAUC"].max():.3f}')
bvo = tt[(tt.Setting == 'Full') & tt.Comparison.str.startswith(best + ' vs') & ~tt.Comparison.str.endswith('Response alone')]
m('nSigBestVsOthers', int((bvo.p < 0.05).sum())); m('nBestVsOthers', len(bvo))

# ---- sensitivity (duplicates kept) ----------------------------------------
sens = cv[cv.Data == 'All 383 rows']
for mo, tag in [('Response alone', 'Resp'), ('LR', 'Lr'), ('RF', 'Rf'), ('SVM', 'Svm')]:
    g = sens[sens.Model == mo]
    m(f'dup{tag}AUC', f3(g.AUC.mean())); m(f'dup{tag}Acc', f3(g.Accuracy.mean()))

# ---- calibration ----------------------------------------------------------
ct = ['\\begin{table}[!htbp]', '\\centering',
      '\\caption{Calibration of the probabilistic models (pooled out-of-fold predictions, first repeat, '
      f'{a["n_dedup"]} unique records). CITL: calibration-in-the-large. Ideal values: slope 1, CITL 0.}}',
      '\\label{tab:calibration}', '\\footnotesize', '\\begin{tabular}{@{}llrrr@{}}', '\\toprule',
      'Features & Model & Brier & Slope & CITL \\\\', '\\midrule']
for fs in ['Full', 'No Response', 'No Response, no Risk']:
    for _, r in cal[cal.Features == fs].iterrows():
        ct.append(f'{fs} & {r.Model} & {r.Brier:.3f} & {r["Calibration slope"]:.2f} & {r["Calibration-in-the-large"]:+.2f} \\\\')
    if fs != 'No Response, no Risk': ct.append('\\addlinespace')
ct += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
(OUT / 'tables/calibration.tex').write_text('\n'.join(ct) + '\n')
for fs, ftag in [('Full', 'Full'), ('No Response', 'NoResp')]:
    for mo, tag in [('LR', 'Lr'), ('RF', 'Rf'), ('DT', 'Dt'), ('MLP', 'Mlp')]:
        r = cal[(cal.Features == fs) & (cal.Model == mo)]
        if len(r):
            r = r.iloc[0]; m(f'cal{tag}{ftag}Slope', f'{r["Calibration slope"]:.2f}'); m(f'cal{tag}{ftag}Brier', f'{r.Brier:.3f}')
            m(f'cal{tag}{ftag}Citl', f'{r["Calibration-in-the-large"]:+.2f}')

# ---- SHAP -----------------------------------------------------------------
for fs, ftag in [('Full', 'Full'), ('No Response', 'NoResp')]:
    s = shp[fs].sort_values(ascending=False)
    s = s[s > 0]
    for i in range(4):
        m(f'shap{ftag}{"ABCD"[i]}', s.index[i]); m(f'shap{ftag}{"ABCD"[i]}Val', f'{s.iloc[i]:.3f}')
m('shapRespRatio', f'{shp.loc["Response", "Full"] / shp["Full"].drop("Response").max():.1f}')

# ---- write macros ---------------------------------------------------------
lines = ['% Auto-generated by make_tables.py -- do not edit by hand.']
for k, v in macros.items():
    assert k.isalpha(), k
    lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
(OUT / 'numbers.tex').write_text('\n'.join(lines) + '\n')
for f in (ROOT / 'figures').glob('fig_*.pdf'):
    shutil.copy(f, OUT / 'figures' / f.name)
print(f'{len(macros)} macros, tables and figures written to {OUT}/')
