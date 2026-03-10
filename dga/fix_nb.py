import json

with open('notebook_classico_dga.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Boxplots: Entropia e Comprimento' in ''.join(cell.get('source', [])):
        cell['source'] = [
            "# Boxplots: Entropia e Comprimento por classe\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "fig.patch.set_facecolor('#0D1117')\n",
            "\n",
            "# Reset index para garantir alinhamento limpo após eventuais NaNs truncados fora\n",
            "df_clean = df.reset_index(drop=True)\n",
            "feat_clean = num_features.reset_index(drop=True)\n",
            "plot_df = feat_clean[['entropy', 'length']].copy()\n",
            "plot_df['label'] = df_clean['isDGA']\n",
            "\n",
            "for ax, feat, title in zip(axes, ['entropy', 'length'], ['Entropia de Shannon', 'Comprimento']):\n",
            "    ax.set_facecolor('#161B22')\n",
            "    \n",
            "    # Extract clean vectors skipping any remaining NaNs safely\n",
            "    d_legit = plot_df[plot_df['label'] == 'legit'][feat].dropna().values\n",
            "    d_dga = plot_df[plot_df['label'] == 'dga'][feat].dropna().values\n",
            "    data_by_class = [d_legit, d_dga]\n",
            "    \n",
            "    bp = ax.boxplot(data_by_class, patch_artist=True,\n",
            "                    medianprops=dict(color='white', linewidth=2),\n",
            "                    flierprops=dict(marker='o', markersize=2, alpha=0.3))\n",
            "    for patch, color in zip(bp['boxes'], [COLORS[0], COLORS[1]]):\n",
            "        patch.set_facecolor(color)\n",
            "        patch.set_alpha(0.7)\n",
            "    ax.set_xticks([1, 2])\n",
            "    ax.set_xticklabels(['Legítimo', 'DGA'], color='white')\n",
            "    ax.set_title(title, color='white', fontsize=13, fontweight='bold')\n",
            "    ax.tick_params(colors='#8B949E')\n",
            "    ax.spines[['top','right','left','bottom']].set_color('#30363D')\n",
            "    ax.set_facecolor('#161B22')\n",
            "\n",
            "fig.suptitle('Features por Classe', color='white', fontsize=15, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('features_by_class.png', bbox_inches='tight', facecolor='#0D1117', dpi=120)\n",
            "plt.show()\n"
        ]
        break

with open('notebook_classico_dga_fixed.ipynb', 'w') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
