"""Planilha de tabelas finais (uma aba por tabela) a partir de resultados_fatorial/.
Uso: python tabelas_finais.py <pasta_resultados> <arquivo_saida.xlsx>"""
import os, sys, pandas as pd
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

RES, SAIDA = sys.argv[1], sys.argv[2]
ler = lambda f: pd.read_csv(os.path.join(RES, f))
abas = {}

abas['T1_autocorrelacao'] = ler('autocorrelacao.csv').rename(columns={'Unnamed: 0': 'variavel'})
eta = pd.read_csv(os.path.join(RES, 'exp1_anova_eta2.csv'), index_col=0)
abas['T2_anova_eta2'] = eta.round(4).reset_index().rename(columns={'index': 'termo'})
e1 = ler('exp1_fatorial.csv')
med = []
for fator in ['protocolo', 'representacao', 'balanceamento', 'modelo']:
    g = e1.groupby(['alvo', fator])[['PR_AUC', 'MCC', 'F1']].mean().round(3).reset_index().rename(columns={fator: 'nivel'})
    g.insert(1, 'fator', fator); med.append(g)
abas['T3_medias_por_fator'] = pd.concat(med, ignore_index=True)
abas['T4_sens_anova'] = ler('sens_anova_eta2.csv')
abas['T5_wilcoxon_pareado'] = ler('sens_wilcoxon_pareado.csv')
abas['T6_balanceamento_3metricas'] = ler('balanceamento_metricas.csv').round(4)
abas['T7_circularidade'] = ler('exp2_circularidade.csv')
if os.path.exists(os.path.join(RES, 'exp8a_hdf_circularidade.csv')):
    abas['T7b_circularidade_HDF'] = ler('exp8a_hdf_circularidade.csv')
abas['T8_ablacao'] = ler('exp3_ablacao_resumo.csv')
r4 = ler('exp4_ruido.csv'); r5 = ler('exp5_faltantes.csv')
if os.path.exists(os.path.join(RES, 'exp8b_hdf_ruido.csv')):
    r4 = pd.concat([r4[r4.alvo != 'HDF'], ler('exp8b_hdf_ruido.csv')]); r5 = pd.concat([r5[r5.alvo != 'HDF'], ler('exp8c_hdf_faltantes.csv')])
abas['T9_ruido'] = r4.groupby(['alvo', 'modelo', 'ruido', 'representacao'])['PR_AUC'].mean().round(3).unstack().reset_index()
abas['T10_faltantes'] = r5.groupby(['alvo', 'modelo', 'falta', 'representacao'])['PR_AUC'].mean().round(3).unstack().reset_index()
abas['T11_ajuste'] = ler('exp6_ajuste_vs_padrao.csv')
c7 = ler('exp7_complexidade.csv'); c7 = c7[(c7.protocolo == 'temporal') & (c7.alvo != 'HDF')]
if os.path.exists(os.path.join(RES, 'exp8d_hdf_complexidade.csv')): c7 = pd.concat([ler('exp8d_hdf_complexidade.csv'), c7])
abas['T12_complexidade'] = c7.groupby(['alvo', 'protocolo', 'representacao', 'profundidade'])['MCC'].mean().round(3).unstack().reset_index()

fino = Side(style='thin', color='999999')
with pd.ExcelWriter(SAIDA, engine='openpyxl') as w:
    for nome, t in abas.items():
        t.to_excel(w, sheet_name=nome[:31], index=False)
        ws = w.sheets[nome[:31]]
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name='Times New Roman', size=10, bold=(c.row == 1))
                c.border = Border(bottom=fino)
                if c.row == 1: c.alignment = Alignment(wrap_text=True, vertical='center')
        for j, col in enumerate(t.columns, 1):
            larg = max([len(str(col))] + [len(str(v)) for v in t.iloc[:, j - 1].head(200)]) + 2
            ws.column_dimensions[get_column_letter(j)].width = min(max(larg, 8), 60)
        ws.freeze_panes = 'A2'
print('planilha salva:', SAIDA, '| abas:', list(abas))
