"""Figuras finais (inglês, 300 dpi) a partir dos CSVs de resultados_fatorial/.
Uso: python figuras_finais.py <pasta_resultados> <pasta_saida> <ai4i2020.csv>"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES, OUT, DADOS = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5,
                     'axes.titleweight': 'bold', 'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COR = {'brutas': '#8C8C8C', 'fisicas': '#2563A8', 'auto': '#D9822B'}
ROT = {'brutas': 'Raw', 'fisicas': 'Physics-informed', 'auto': 'Automatic (no domain knowledge)'}
ALVO = {'failure': 'Any failure', 'HDF': 'HDF', 'PWF': 'PWF', 'OSF': 'OSF', 'TWF': 'TWF'}
ler = lambda f: pd.read_csv(os.path.join(RES, f))
def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome + '.png')); fig.savefig(os.path.join(OUT, nome + '.pdf')); plt.close(fig)

# ---------- Fig. 1: dependência temporal e episódio de HDF ----------
d = pd.read_csv(DADOS).sort_values('UDI')
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(d['UDI'], d['Air temperature [K]'], color='#8C8C8C', lw=0.6, label='Air temperature')
ax.plot(d['UDI'], d['Process temperature [K]'], color='#2563A8', lw=0.6, label='Process temperature')
h = d[d['HDF'] == 1]
ax.scatter(h['UDI'], h['Air temperature [K]'], s=6, color='#C0392B', zorder=3, label='HDF failure (n = 115)')
ax.axvspan(h['UDI'].min(), h['UDI'].max(), color='#C0392B', alpha=0.08)
for k, c in enumerate([5000, 6000, 7000, 8000]):
    ax.axvline(c, color='k', lw=0.6, ls=':', label='Temporal split points' if k == 0 else None)
ax.set_xlabel('Record order (UDI)'); ax.set_ylabel('Temperature (K)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), fontsize=7.5, ncol=4)
salvar(fig, 'fig1_temporal_dependence')

# ---------- Fig. 2: por que a representação importa (PWF) ----------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
t, r, y = d['Torque [Nm]'], d['Rotational speed [rpm]'], d['PWF']
P = t * r * 2 * np.pi / 60
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
fig.subplots_adjust(wspace=0.25)
X = np.c_[r, t]; dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=0).fit(X, y)
gr, gt = np.meshgrid(np.linspace(1150, 2900, 400), np.linspace(2, 80, 400))
Z = dt.predict(np.c_[gr.ravel(), gt.ravel()]).reshape(gr.shape)
a = ax[0]; a.contourf(gr, gt, Z, levels=[-.5, .5, 1.5], colors=['white', '#F6D5B5'], alpha=.9)
a.scatter(r[y == 0], t[y == 0], s=1.5, c='#B0B0B0', alpha=.35, rasterized=True)
a.scatter(r[y == 1], t[y == 1], s=6, c='#C0392B', label='PWF')
rr = np.linspace(1150, 2900, 300)
for p in (3500, 9000): a.plot(rr, p / (rr * 2 * np.pi / 60), '--', c='#2563A8', lw=1)
a.set_xlim(1150, 2900); a.set_ylim(2, 80); a.set_xlabel('Rotational speed (rpm)'); a.set_ylabel('Torque (N·m)')
a.set_title(f'(a) Raw variables\ndepth-3 tree, MCC = {matthews_corrcoef(y, dt.predict(X)):.2f}')
b = ax[1]; j = np.random.default_rng(0).uniform(-.4, .4, len(P))
b.axvspan(P.min() - 200, 3500, color='#F6D5B5', alpha=.6, zorder=0); b.axvspan(9000, P.max() + 200, color='#F6D5B5', alpha=.6, zorder=0)
b.scatter(P[y == 0], j[y == 0], s=1.5, c='#B0B0B0', alpha=.35, zorder=2, rasterized=True)
b.scatter(P[y == 1], j[y == 1], s=6, c='#C0392B', zorder=3)
for p in (3500, 9000): b.axvline(p, c='#2563A8', ls='--', lw=1, zorder=4)
dt2 = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=0).fit(P.values.reshape(-1, 1), y)
b.set_yticks([]); b.spines['left'].set_visible(False); b.set_xlabel('Power = torque × ω (W)')
b.set_title(f'(b) Physics-informed feature\ndepth-2 tree, MCC = {matthews_corrcoef(y, dt2.predict(P.values.reshape(-1, 1))):.2f}')
salvar(fig, 'fig2_representation_pwf')

# ---------- Fig. 3: η² por fator e alvo ----------
eta = pd.read_csv(os.path.join(RES, 'exp1_anova_eta2.csv'), index_col=0)
termos = ['protocolo', 'representacao', 'balanceamento', 'modelo', 'representacao:modelo', 'Residual']
rot_t = {'protocolo': 'Protocol', 'representacao': 'Representation', 'balanceamento': 'Class balancing',
         'modelo': 'Model', 'representacao:modelo': 'Representation × model', 'Residual': 'Residual'}
alvos = [a for a in ALVO if f'{a} η²' in eta.columns]
fig, ax = plt.subplots(figsize=(7, 2.8))
largura = 0.8 / len(termos); paleta = ['#B07AA1', '#2563A8', '#59A14F', '#8C8C8C', '#76B7B2', '#E0E0E0']
for k, termo in enumerate(termos):
    v = [eta.loc[termo, f'{a} η²'] if termo in eta.index else np.nan for a in alvos]
    ax.bar(np.arange(len(alvos)) + (k - len(termos) / 2 + .5) * largura, v, largura, label=rot_t[termo],
           color=paleta[k], edgecolor='white', linewidth=.4)
ax.set_xticks(range(len(alvos))); ax.set_xticklabels([ALVO[a] + ('*' if a == 'HDF' else '') for a in alvos])
ax.set_ylabel('η² (share of PR-AUC variance)'); ax.set_ylim(0, 1)
ax.legend(ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, 1.18))
ax.text(1, -0.16, '*HDF: random split only', transform=ax.transAxes, ha='right', fontsize=7)
salvar(fig, 'fig3_variance_decomposition')

# ---------- Fig. 4: efeito da representação por protocolo ----------
e1 = ler('exp1_fatorial.csv')
fig, axs = plt.subplots(1, len(alvos), figsize=(7.2, 2.2), sharey=True)
for ax, a in zip(axs, alvos):
    g = e1[e1.alvo == a].groupby(['protocolo', 'representacao'])['PR_AUC'].agg(['mean', 'sem'])
    for k, (prot, mk) in enumerate([('aleatorio', 'o'), ('temporal', 's')]):
        if prot not in g.index.get_level_values(0): continue
        for i, rep in enumerate(['brutas', 'auto', 'fisicas']):
            m, s = g.loc[(prot, rep)]
            ax.errorbar(i + (k - .5) * .25, m, yerr=1.96 * s, fmt=mk, color=COR[rep], ms=4, capsize=1.5,
                        mfc=COR[rep] if prot == 'temporal' else 'white')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Raw', 'Auto', 'Phys.'], fontsize=7.5)
    ax.set_title(ALVO[a]); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=.3)
axs[0].set_ylabel('Mean PR-AUC (95% CI)')
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker='o', ls='', color='k', mfc='white'), Line2D([], [], marker='s', ls='', color='k')],
           ['Random split', 'Temporal split'], ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.08), fontsize=7.5)
salvar(fig, 'fig4_representation_by_protocol')

# ---------- Fig. 5: comparações pareadas ----------
pw = ler('sens_wilcoxon_pareado.csv')
comps = ['fisicas − brutas', 'auto − brutas', 'fisicas − auto', 'smote − nenhum', 'class_weight − nenhum',
         'Gradient Boosting − Random Forest', 'Random Forest − Árvore rasa']
rot_c = {'fisicas − brutas': 'Physics − raw', 'auto − brutas': 'Automatic − raw', 'fisicas − auto': 'Physics − automatic',
         'smote − nenhum': 'SMOTE − none', 'class_weight − nenhum': 'Class weight − none',
         'Gradient Boosting − Random Forest': 'GB − RF', 'Random Forest − Árvore rasa': 'RF − shallow tree'}
mk = {'failure': 'o', 'HDF': 'D', 'PWF': 's', 'OSF': '^', 'TWF': 'v'}
cor_a = {'failure': '#000000', 'HDF': '#B07AA1', 'PWF': '#2563A8', 'OSF': '#D9822B', 'TWF': '#8C8C8C'}
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for i, c in enumerate(comps):
    for k, a in enumerate(alvos):
        rr = pw[(pw['comparação'] == c) & (pw.alvo == a)]
        if rr.empty: continue
        v = rr['diferença mediana'].iloc[0]; sig = rr['significativo (5%)'].iloc[0] == 'sim'
        ax.scatter(v, i + (k - 2) * .12, marker=mk[a], s=22, color=cor_a[a] if sig else 'white',
                   edgecolor=cor_a[a], zorder=3, label=ALVO[a] if i == 0 else None)
ax.axvline(0, color='k', lw=.7)
ax.set_yticks(range(len(comps))); ax.set_yticklabels([rot_c[c] for c in comps]); ax.invert_yaxis()
ax.set_xlabel('Median paired difference in PR-AUC'); ax.grid(axis='x', alpha=.3)
ax.legend(fontsize=7.5, loc='lower right', title='Filled = significant (Holm, 5%)', title_fontsize=7)
salvar(fig, 'fig5_paired_comparisons')

# ---------- Fig. 6: robustez (temporal + HDF aleatório) ----------
def junta(f_temp, f_hdf, chave):
    t = ler(f_temp); t = t[t.alvo != 'HDF']
    if os.path.exists(os.path.join(RES, f_hdf)): t = pd.concat([t, ler(f_hdf)])
    return t
rob = [(junta('exp4_ruido.csv', 'exp8b_hdf_ruido.csv', 'ruido'), 'ruido', 'Noise (× SD)'),
       (junta('exp5_faltantes.csv', 'exp8c_hdf_faltantes.csv', 'falta'), 'falta', 'Missing-value rate')]
fig, axs = plt.subplots(2, len(alvos), figsize=(7.4, 3.8), sharey=True)
for li, (res, chave, xl) in enumerate(rob):
    for ax, a in zip(axs[li], alvos):
        g = res[(res.alvo == a) & (res.modelo == 'Random Forest')].groupby([chave, 'representacao'])['PR_AUC'].mean().unstack()
        for rep in ['brutas', 'auto', 'fisicas']:
            if rep in g: ax.plot(g.index, g[rep], 'o-', ms=2.5, lw=1.1, color=COR[rep], label=ROT[rep])
        ax.set_title(ALVO[a] + ('*' if a == 'HDF' else '')); ax.set_xlabel(xl, fontsize=7.5); ax.grid(alpha=.3)
        ax.set_ylim(0, 1.05); ax.tick_params(labelsize=7)
    axs[li][0].set_ylabel('PR-AUC (Random Forest)')
h,l=axs[0][0].get_legend_handles_labels(); fig.legend(h,l,ncol=3,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.04))
fig.text(.5, -.08, '*HDF: random split; other targets: temporal split', ha='center', fontsize=7)
fig.tight_layout(); salvar(fig, 'fig6_robustness')

# ---------- Fig. 7: complexidade necessária ----------
c7 = ler('exp7_complexidade.csv'); c7 = c7[(c7.protocolo == 'temporal') & (c7.alvo != 'HDF')]
if os.path.exists(os.path.join(RES, 'exp8d_hdf_complexidade.csv')): c7 = pd.concat([ler('exp8d_hdf_complexidade.csv'), c7])
modos = [a for a in ['HDF', 'PWF', 'OSF', 'TWF'] if a in c7.alvo.unique()]
fig, axs = plt.subplots(1, len(modos), figsize=(7.2, 2.2), sharey=True)
for ax, a in zip(axs, modos):
    g = c7[c7.alvo == a].groupby(['profundidade', 'representacao'])['MCC'].mean().unstack()
    for rep in ['brutas', 'fisicas']:
        ax.plot(g.index, g[rep], 'o-', ms=3, lw=1.2, color=COR[rep], label=ROT[rep])
    ax.set_title(ALVO[a] + ('*' if a == 'HDF' else '')); ax.set_xlabel('Tree depth'); ax.set_xticks(range(1,9)); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.3)
axs[0].set_ylabel('Test MCC'); axs[0].legend(fontsize=7, loc='center right')
salvar(fig, 'fig7_required_complexity')
print('figuras salvas em', OUT, sorted(os.listdir(OUT)))
