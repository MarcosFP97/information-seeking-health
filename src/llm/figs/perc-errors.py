import matplotlib.pyplot as plt
import numpy as np

# Datos
categories = ['Incorrect medical consensus', 'Misinterpretation of the question', 'Ambiguous Answer']
no_context = [75, 25,  0]  # Porcentajes de errores para 'no context'
expert = [57, 14,  29]      # Porcentajes de errores para 'expert'

x = np.arange(len(categories))  # Posición en el eje x
width = 0.35  # Ancho de las barras

# Crear el gráfico
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, no_context, width, label='no context', color='steelblue')
bars2 = ax.bar(x + width/2, expert, width, label='expert', color='cornflowerblue')

# Etiquetas y título
ax.set_ylabel('%s of total errors', fontsize=16)
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=16)
ax.legend(fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mostrar gráfico
plt.xticks(rotation=45)
plt.yticks(fontsize=16)
plt.ylim(0,100)
plt.savefig('perc_errors.png', bbox_inches='tight')
plt.show()