import pandas as pd
pd.set_option('display.precision', 2)
pd.options.display.float_format = lambda x : '{:.0f}'.format(x) if int(x) == x else '{:,.2f}'.format(x)

df = pd.read_csv('results.csv', index_col=0)
df = df.transpose()


not_entangled_f1_score = df['not_entangled_f1-score']
entangled_f1_score = df['entangled_f1-score']
df['macro_f1'] = (1 / 2) * (not_entangled_f1_score + entangled_f1_score)
df = df.sort_values(by='macro_f1', ascending=False)
df = df.transpose()

df = df.round(decimals=2)

df.to_csv('results2.csv')

replace_pairs = [
    ("_", " "),
    ("artisanal", "Artisanal"),
    ("resnet", "ResNet-18"),
    ("alexnet", "Modified AlexNet"),
    ("mlp", "Custom MLP"),
    ("SVM", "Support Vector Machine"),
]
cs = ["c" for c in df.columns]
cs.append("c")
latex_table = df.to_latex(caption='Results Summary', label='tab:summary', column_format=f'|{"|".join(cs)}|')
lines = "\n".join(latex_table.split("\n")[3:-2])
for target, solution in replace_pairs:
    lines = lines.replace(target, solution)
    lines = lines.replace("000", "")

print(lines)
with open('tex/results_table.tex', 'w') as results_table:
    results_table.write(lines)  # get rid of unnecessary things

print(df)