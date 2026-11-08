def anova_alvo(d, metrica='PR_AUC'):
    # ANOVA tipo II. A repetição (1ª a 4ª divisão) entra como bloco cruzado com o protocolo,
    # e não aninhado nele, para não confundir o efeito do protocolo. Fatores com um único nível
    # (ex.: HDF sem casos no teste temporal) são retirados automaticamente.
    d = d.dropna(subset=[metrica]).copy()
    # número da repetição dentro de cada protocolo (cortes temporais e sementes viram 1..4)
    d['repeticao'] = d.groupby('protocolo')['divisao'].transform(lambda s: s.astype('category').cat.codes + 1)
    fatores = [f for f in ['protocolo', 'representacao', 'balanceamento', 'modelo'] if d[f].nunique() > 1]
    termos = [f'C({f})' for f in fatores]
    pares = [('protocolo', 'representacao'), ('protocolo', 'modelo'), ('representacao', 'balanceamento'),
             ('representacao', 'modelo'), ('balanceamento', 'modelo')]
    termos += [f'C({a}):C({b})' for a, b in pares if a in fatores and b in fatores]
    if d['repeticao'].nunique() > 1:
        termos.append('C(repeticao)')
    a = anova_lm(smf.ols(f'{metrica} ~ ' + ' + '.join(termos), d).fit(), typ=2)
    a['eta2'] = a['sum_sq'] / a['sum_sq'].sum()
    return a

tabelas = []
for alvo in ALVOS:
    d = exp1[exp1.alvo == alvo]
    if d['PR_AUC'].notna().sum() < 30:
        print(f'{alvo}: poucas avaliações válidas, ANOVA não calculada'); continue
    if d.dropna(subset=['PR_AUC'])['protocolo'].nunique() < 2:
        print(f'{alvo}: só um protocolo tem casos positivos no teste; o efeito do protocolo não é estimado')
    a = anova_alvo(d)
    a = a.drop(index=[i for i in a.index if 'repeticao' in i])
    t = a[['eta2', 'PR(>F)']].rename(index=lambda s: s.replace('C(', '').replace(')', ''))
    t.columns = [f'{alvo} η²', f'{alvo} p']
    tabelas.append(t)
eta = pd.concat(tabelas, axis=1)
eta.to_csv(os.path.join(SAIDA, 'exp1_anova_eta2.csv'))
display(eta.round(3))

# Figura: η² dos efeitos principais e interações, por alvo
cols = [c for c in eta.columns if 'η²' in c]
ax = eta[cols].drop(index='Residual').plot.barh(figsize=(8, 6), width=0.8)
ax.set_xlabel('η² (parcela da variação do PR-AUC)'); ax.set_title('Exp. 1 — quanto cada fator explica')
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(os.path.join(SAIDA, 'fig_exp1_eta2.png'), dpi=200); plt.show()
