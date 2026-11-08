# ================================================================
# BALANCEAMENTO: A CONCLUSÃO SE MANTÉM COM MÉTRICAS QUE DEPENDEM DO LIMIAR?
# O PR-AUC mede ordenação; o balanceamento atua sobretudo na calibração e no limiar.
# Aqui as mesmas comparações pareadas são repetidas com MCC e F1, calculados no limiar
# escolhido na validação (lembrete: o limiar é otimizado por F1 em cada combinação).
# Autocontida: lê resultados_fatorial/exp1_fatorial.csv.
# ================================================================
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

SAIDA = 'resultados_fatorial'
exp1 = pd.read_csv(os.path.join(SAIDA, 'exp1_fatorial.csv'))
ALVOS = [a for a in ['failure', 'HDF', 'PWF', 'OSF', 'TWF'] if a in exp1.alvo.unique()]
FATORES = ['protocolo', 'representacao', 'balanceamento', 'modelo']
METRICAS = ['PR_AUC', 'MCC', 'F1']

def pareado(d, alvo, fator, a, b, metrica):
    d = d[(d.alvo == alvo) & d[metrica].notna()]
    chaves = ['divisao'] + [f for f in FATORES if f != fator]
    w = d.pivot_table(index=chaves, columns=fator, values=metrica).dropna(subset=[a, b])
    dif = w[b] - w[a]
    p = wilcoxon(dif).pvalue if (dif != 0).any() else 1.0
    return {'alvo': alvo, 'métrica': metrica, 'comparação': f'{b} − {a}', 'n': len(dif),
            'dif. mediana': dif.median(), 'dif. média': dif.mean(),
            '% pares B > A': 100 * (dif > 0).mean(), 'p': p}

comps = [('nenhum', 'smote'), ('nenhum', 'class_weight'), ('class_weight', 'smote')]

# ---------- 1. Todos os modelos juntos ----------
linhas = [pareado(exp1, alvo, 'balanceamento', a, b, m) for m in METRICAS for alvo in ALVOS for a, b in comps]
res = pd.DataFrame(linhas)
res['p (Holm)'] = multipletests(res['p'], method='holm')[1]
res['sig.'] = np.where(res['p (Holm)'] < 0.05, '✓', '')

def resumo(t):
    t = t.copy()
    t['célula'] = t.apply(lambda r: f"{r['dif. mediana']:+.3f} ({r['% pares B > A']:.0f}%) {r['sig.']}", axis=1)
    return t.pivot_table(index=['comparação', 'métrica'], columns='alvo', values='célula', aggfunc='first')[ALVOS]

print('1) Balanceamento — todos os modelos juntos')
print('   célula = diferença mediana (% de pares em que o 1º nível supera o 2º), ✓ = significativo após Holm')
display(resumo(res))
res.to_csv(os.path.join(SAIDA, 'balanceamento_metricas.csv'), index=False)

# ---------- 2. Por modelo (o efeito do balanceamento depende do modelo?) ----------
lin2 = []
for mod in sorted(exp1.modelo.unique()):
    sub = exp1[exp1.modelo == mod]
    for m in ['MCC', 'PR_AUC']:
        for alvo in ALVOS:
            for a, b in comps:
                r = pareado(sub, alvo, 'balanceamento', a, b, m); r['modelo'] = mod; lin2.append(r)
res2 = pd.DataFrame(lin2)
res2['p (Holm)'] = multipletests(res2['p'], method='holm')[1]
res2['sig.'] = np.where(res2['p (Holm)'] < 0.05, '✓', '')
res2.to_csv(os.path.join(SAIDA, 'balanceamento_por_modelo.csv'), index=False)
print('\n2) MCC: SMOTE − nenhum e class_weight − nenhum, por modelo')
t2 = res2[(res2['métrica'] == 'MCC') & res2['comparação'].isin(['smote − nenhum', 'class_weight − nenhum'])].copy()
t2['célula'] = t2.apply(lambda r: f"{r['dif. mediana']:+.3f} ({r['% pares B > A']:.0f}%) {r['sig.']}", axis=1)
display(t2.pivot_table(index=['modelo', 'comparação'], columns='alvo', values='célula', aggfunc='first')[ALVOS])

# ---------- 3. Médias por nível ----------
print('\n3) Médias por balanceamento (todos os modelos)')
display(exp1.groupby(['alvo', 'balanceamento'])[METRICAS].mean().round(3).unstack('balanceamento'))

# ---------- 4. Veredito automático ----------
print('\n4) Veredito: em quantos alvos cada técnica é significativamente melhor / pior que não balancear')
for m in METRICAS:
    for comp in ['smote − nenhum', 'class_weight − nenhum']:
        t = res[(res['métrica'] == m) & (res['comparação'] == comp)]
        melhor = t[(t['sig.'] == '✓') & (t['dif. mediana'] > 0)].alvo.tolist()
        pior = t[(t['sig.'] == '✓') & (t['dif. mediana'] < 0)].alvo.tolist()
        print(f'  {m:6s} {comp:24s} melhor em {melhor or "nenhum"} | pior em {pior or "nenhum"}')
