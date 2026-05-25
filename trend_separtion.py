import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# Настройка шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. Загрузка данных ==========
yield_df = pd.read_csv('yield.csv')
yield_df['year'] = yield_df['year'].astype(int)
# Перевод из центнеров на гектар (ц/га) в тонны на гектар (т/га)
yield_df['yield_per_hectare'] = yield_df['yield_per_hectare'] * 10.0

#==========SPEI вегетационного сезона (май-сентябрь, единый подход)==========
spei_monthly = pd.read_csv('jilin_spei_monthly.csv')
spei_filtered = spei_monthly[(spei_monthly['year'] >= 1980) &
                              (spei_monthly['year'] <= 2020)].copy()
spei_growing = spei_filtered[spei_filtered['month'].between(5, 9)].groupby('year')['SPEI'].mean().reset_index()
spei_growing.columns = ['year', 'SPEI_growing_season_new']

df = yield_df.merge(spei_growing, on='year', how='inner')
df = df.reset_index(drop=True)

spei_col = 'SPEI_growing_season_new'
yield_col = 'yield_per_hectare'

print("Данные загружены:")
print(f"  Всего лет: {len(df)}")
print(f"  Диапазон годов: {df['year'].min()} - {df['year'].max()}")
print(f"Используемый столбец SPEI: {spei_col}")
print(f"\nПервые 5 строк:\n{df.head()}")

# Сопоставление столбцов
spei_col = 'SPEI_growing_season_new'  # Средний SPEI за вегетационный сезон
yield_col = 'yield_per_hectare'   # Урожайность кукурузы

# ========== 2. Выделение трендовой урожайности (5-летнее скользящее среднее) ==========
window = 5
df['trend_yield'] = df[yield_col].rolling(window=window, center=True).mean()

# Обработка граничных значений
for i in range(len(df)):
    if pd.isna(df.loc[i, 'trend_yield']):
        start = max(0, i-2)
        end = min(len(df), i+3)
        available = df.loc[start:end, yield_col].dropna()
        if len(available) >= 3:
            df.loc[i, 'trend_yield'] = available.mean()
df['trend_yield'] = df['trend_yield'].interpolate()
            
# ========== 3. Расчёт метеорологической урожайности ==========
df['weather_yield'] = df[yield_col] - df['trend_yield']
df['relative_weather_yield'] = (df['weather_yield'] / df['trend_yield']) * 100

# ========== 4. Корреляционный анализ ==========
print("\n" + "="*60)
print("Корреляционный анализ после выделения тренда")
print("="*60)

def significance_label(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return 'ns'
    
# 4.1 Абсолютная метеорологическая урожайность vs SPEI
r1, p1 = stats.pearsonr(df['weather_yield'], df[spei_col])
print(f"\n① Абсолютная метеорологическая урожайность vs SPEI вегетационного сезона:")
print(f"   r = {r1:.4f}, p = {p1:.4f}{significance_label(p1)}")

# 4.2 Относительная метеорологическая урожайность vs SPEI
r2, p2 = stats.pearsonr(df['relative_weather_yield'], df[spei_col])
print(f"\n② Относительная метеорологическая урожайность vs SPEI вегетационного сезона:")
print(f"   r = {r2:.4f}, p = {p2:.4f}{significance_label(p2)}")

# 4.3 Трендовая урожайность vs Год
r3, p3 = stats.pearsonr(df['year'], df['trend_yield'])
print(f"\n③ Трендовая урожайность vs Год:")
print(f"   r = {r3:.4f}, p = {p3:.4f}")

# 4.4 Сравнение: исходная урожайность vs SPEI (до выделения тренда)
r0, p0 = stats.pearsonr(df[yield_col], df[spei_col])
print(f"\n④ [Сравнение] Исходная урожайность vs SPEI (до выделения тренда):")
print(f"   r = {r0:.4f}, p = {p0:.4f}")

# ========== 5. Визуализация ==========
# ========== 5. Визуализация ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Рис. (a): Фактическая, трендовая и метеорологическая урожайность
ax1 = axes[0, 0]
ax1.plot(df['year'], df[yield_col], 'o-', color='blue', 
         label='Фактическая урожайность', markersize=4)
ax1.plot(df['year'], df['trend_yield'], '-', color='red', linewidth=2, 
         label='Трендовая урожайность (5-летнее скользящее)')
ax1.fill_between(df['year'], df['trend_yield'], df[yield_col],
                  where=df[yield_col] >= df['trend_yield'], 
                  color='green', alpha=0.3, 
                  label='Метеорологическая урожайность > 0')
ax1.fill_between(df['year'], df['trend_yield'], df[yield_col],
                  where=df[yield_col] < df['trend_yield'], 
                  color='orange', alpha=0.3, 
                  label='Метеорологическая урожайность < 0')
ax1.set_xlabel('Год', fontsize=10)
ax1.set_ylabel('Урожайность (т/га)', fontsize=10)
ax1.set_title('Выделение тренда урожайности кукурузы', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Рис. (b): Относительная метеорологическая урожайность
ax2 = axes[0, 1]
colors = ['red' if x < 0 else 'green' for x in df['relative_weather_yield']]
ax2.bar(df['year'], df['relative_weather_yield'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('Год', fontsize=10)
ax2.set_ylabel('Относительная метеорологическая\nурожайность (%)', fontsize=10)
ax2.set_title('Изменение относительной\nметеорологической урожайности', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Рис. (c): Диаграмма рассеяния
ax3 = axes[1, 0]
sc = ax3.scatter(df[spei_col], df['relative_weather_yield'],
            c=df['year'], cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

z = np.polyfit(df[spei_col], df['relative_weather_yield'], 1)
p_trend = np.poly1d(z)
x_line = np.linspace(df[spei_col].min(), df[spei_col].max(), 100)
ax3.plot(x_line, p_trend(x_line), 'r-', linewidth=1.5, 
         label=f'Линия тренда (r = {r2:.3f}, p = {p2:.4f})')
ax3.set_xlabel('Средний SPEI вегетационного сезона', fontsize=10)
ax3.set_ylabel('Относительная метеорологическая\nурожайность (%)', fontsize=10)
ax3.set_title('Относительная метеорологическая\nурожайность vs SPEI', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8, loc='lower right')
cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Год', fontsize=9)
ax3.grid(True, alpha=0.3)

# Рис. (d): Временной ряд SPEI
ax4 = axes[1, 1]
ax4.plot(df['year'], df[spei_col], 'o-', color='darkblue', linewidth=1.5, markersize=4)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
drought = df[df[spei_col] < -1]
ax4.scatter(drought['year'], drought[spei_col], color='red', s=50, zorder=5, 
           label=f'Засушливые годы (SPEI < -1): {len(drought)}')
ax4.fill_between(df['year'], 0, df[spei_col],
                  where=df[spei_col] < 0, color='orange', alpha=0.3)
ax4.set_xlabel('Год', fontsize=10)
ax4.set_ylabel('SPEI', fontsize=10)
ax4.set_title('Временной ряд SPEI\nвегетационного сезона', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8, loc='lower left')
ax4.grid(True, alpha=0.3)

# 添加子图标签 (a), (b), (c), (d)
labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flat, labels):
    ax.text(0.03, 0.94, label, transform=ax.transAxes, fontsize=13, 
            fontweight='bold', va='top', ha='left')

# 调整布局 - 关键参数
plt.subplots_adjust(
    left=0.10, right=0.93,
    top=0.93, bottom=0.07,
    wspace=0.50, hspace=0.45
)

plt.savefig('trend_separation_results.png', dpi=300, bbox_inches='tight')
print("Результаты сохранены в trend_separation_results.png")
plt.show()