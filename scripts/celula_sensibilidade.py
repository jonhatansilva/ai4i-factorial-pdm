# ================================================================
# ANÁLISES DE SENSIBILIDADE DO EXPERIMENTO 1
# Autocontida: lê resultados_fatorial/exp1_fatorial.csv (não refaz nenhum modelo).
#   S1. ANOVA original (referência)
#   S2. ANOVA sem a regressão logística (ela infla o peso do fator "modelo")
#   S3. ANOVA com PR-AUC em escala logit (corrige a saturação perto de 1)
#   S4. Comparações par a par: Tukey HSD e Wilcoxon pareado com correção de Holm
# ================================================================
import os, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

SAIDA = 'resultados_fatorial'
exp1 = pd.read_csv(os.path.join(SAIDA, 'exp1_fatorial.csv'))
ALVOS = [a for a in ['failure', 'HDF', 'PWF', 'OSF', 'TWF'] if a in exp1.alvo.unique()]
FATORES = ['protocolo', 'representacao', 'balanceamento', 'modelo']

def anova_alvo(d, metrica='PR_AUC'):
    d = d.dropna(subset=[metrica]).copy()
    d['repeticao'] = d.groupby('protocolo')['divisao'].transform(lambda s: s.astype('category').cat.codes + 1)
    fat = [f for f in FATORES if d[f].nunique() > 1]
    termos = [f'C({f})' for f in fat]
    pares = [('protocolo', 'representacao'), ('protocolo', 'modelo'), ('representacao', 'balanceamento'),
             ('representacao', 'modelo'), ('balanceamento', 'modelo')]
    termos += [f'C({a}):C({b})' for a, b in pares if a in fat and b in fat]
    if d['repeticao'].nunique() > 1:
        termos.append('C(repeticao)')
    a = anova_lm(smf.ols(f'{metrica} ~ ' + ' + '.join(termos), d).fit(), typ=2)
    a['eta2'] = a['sum_sq'] / a['sum_sq'].sum()
    a = a.drop(index=[i for i in a.index if 'repeticao' in i])
    return a.rename(index=lambda s: s.replace('C(', '').replace(')', ''))

exp1['PR_AUC_logit'] = np.log(exp1['PR_AUC'].clip(0.001, 0.999) / (1 - exp1['PR_AUC'].clip(0.001, 0.999)))

variantes = {
    'S1 original':      (exp1, 'PR_AUC'),
    'S2 sem logística': (exp1[exp1.modelo != 'Logística'], 'PR_AUC'),
    'S3 logit':         (exp1, 'PR_AUC_logit'),
    'S2+S3':            (exp1[exp1.modelo != 'Logística'], 'PR_AUC_logit'),
}

# ---------------- S1–S3: η² dos efeitos principais e da interação representação × modelo ----------------
linhas = []
for alvo in ALVOS:
    for nome, (d, met) in variantes.items():
        dd = d[d.alvo == alvo]
        if dd[met].notna().sum() < 30: continue
        a = anova_alvo(dd, met)
        lin = {'alvo': alvo, 'variante': nome}
        for termo in ['protocolo', 'representacao', 'balanceamento', 'modelo', 'representacao:modelo', 'Residual']:
            lin[termo] = round(a.loc[termo, 'eta2'], 3) if termo in a.index else np.nan
        # fator principal dominante
        principais = {k: lin[k] for k in ['protocolo', 'representacao', 'balanceamento', 'modelo'] if pd.notna(lin[k])}
        lin['fator dominante'] = max(principais, key=principais.get)
        linhas.append(lin)
sens = pd.DataFrame(linhas)
sens.to_csv(os.path.join(SAIDA, 'sens_anova_eta2.csv'), index=False)
print('η² por variante da ANOVA (efeitos principais e interação representação × modelo)')
display(sens)

print('\nO fator dominante muda entre as variantes?')
display(sens.pivot(index='alvo', columns='variante', values='fator dominante'))

# ---------------- S4a: Tukey HSD ----------------
# Atenção: o Tukey trata as observações como independentes; o teste pareado (S4b) é o mais adequado
# a este desenho, em que cada nível é comparado nas mesmas condições.
tk = []
for alvo in ALVOS:
    d = exp1[(exp1.alvo == alvo) & exp1.PR_AUC.notna()]
    for fator in ['representacao', 'balanceamento', 'modelo']:
        r = pairwise_tukeyhsd(d['PR_AUC'], d[fator])
        t = pd.DataFrame(r.summary().data[1:], columns=r.summary().data[0])
        t.insert(0, 'fator', fator); t.insert(0, 'alvo', alvo)
        tk.append(t)
tukey = pd.concat(tk, ignore_index=True)
tukey.to_csv(os.path.join(SAIDA, 'sens_tukey.csv'), index=False)

# ---------------- S4b: Wilcoxon pareado + Holm ----------------
# Cada comparação usa pares que diferem só no fator testado (mesmo alvo, protocolo, divisão e demais fatores).
def pareado(alvo, fator, a, b):
    d = exp1[(exp1.alvo == alvo) & exp1.PR_AUC.notna()]
    chaves = ['divisao'] + [f for f in FATORES if f != fator]
    w = d.pivot_table(index=chaves, columns=fator, values='PR_AUC').dropna(subset=[a, b])
    dif = w[b] - w[a]
    p = wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
    return {'alvo': alvo, 'fator': fator, 'comparação': f'{b} − {a}', 'n pares': len(dif),
            'diferença mediana': round(dif.median(), 3), 'diferença média': round(dif.mean(), 3),
            '% pares em que B > A': round(100 * (dif > 0).mean(), 1), 'p': p}

comparacoes = [('representacao', 'brutas', 'fisicas'), ('representacao', 'brutas', 'auto'),
               ('representacao', 'auto', 'fisicas'),
               ('balanceamento', 'nenhum', 'smote'), ('balanceamento', 'nenhum', 'class_weight'),
               ('balanceamento', 'class_weight', 'smote'),
               ('modelo', 'Random Forest', 'Gradient Boosting'), ('modelo', 'Árvore rasa', 'Random Forest')]
pw = pd.DataFrame([pareado(alvo, *c) for alvo in ALVOS for c in comparacoes])
pw['p (Holm)'] = multipletests(pw['p'], method='holm')[1]
pw['significativo (5%)'] = np.where(pw['p (Holm)'] < 0.05, 'sim', 'não')
pw['p'] = pw['p'].map(lambda v: f'{v:.2g}'); pw['p (Holm)'] = pw['p (Holm)'].map(lambda v: f'{v:.2g}')
pw.to_csv(os.path.join(SAIDA, 'sens_wilcoxon_pareado.csv'), index=False)
print('\nComparações pareadas (Wilcoxon, correção de Holm sobre todas as comparações)')
pd.set_option('display.max_rows', 100); pd.set_option('display.width', 200)
display(pw)

print('\nArquivos salvos: sens_anova_eta2.csv, sens_tukey.csv, sens_wilcoxon_pareado.csv')
