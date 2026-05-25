import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Настройка шрифта (поддержка кириллицы)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("Шаг 3: Анализ влияния засухи по фазам развития кукурузы")
print("="*60)

# ========== 1. Загрузка данных ==========
spei_monthly = pd.read_csv('jilin_spei_monthly.csv')
yield_df = pd.read_csv('yield.csv')

print(f"\nДанные SPEI (длинный формат): {spei_monthly.shape[0]} строк × {spei_monthly.shape[1]} столбцов")
print(f"Столбцы: {spei_monthly.columns.tolist()}")
print(f"Первые 5 строк:\n{spei_monthly.head()}")
print(f"Диапазон годов: {spei_monthly['year'].min()} - {spei_monthly['year'].max()}")
print(f"Диапазон месяцев: {sorted(spei_monthly['month'].unique())}")

# ========== 2. Преобразование в широкий формат ==========
spei_filtered = spei_monthly[(spei_monthly['year'] >= 1980) & 
                              (spei_monthly['year'] <= 2020)].copy()

spei_wide = spei_filtered.pivot(index='year', columns='month', values='SPEI')
spei_wide.columns = [f'SPEI_{int(col):02d}' for col in spei_wide.columns]
spei_wide = spei_wide.reset_index()

print(f"\nПосле преобразования: {spei_wide.shape[0]} строк × {spei_wide.shape[1]} столбцов")
print(f"Столбцы: {spei_wide.columns.tolist()}")
print(f"Первые 3 строки:\n{spei_wide.head(3)}")

# ========== 3. Определение фаз развития кукурузы ==========
growth_stages = {
    'Весь сезон': ['SPEI_05', 'SPEI_06', 'SPEI_07', 'SPEI_08', 'SPEI_09'],
    'Посев-всходы': ['SPEI_05'],
    'Всходы-вытягивание': ['SPEI_05', 'SPEI_06'],        
    'Вытягивание-цветение': ['SPEI_06', 'SPEI_07'],     
    'Цветение-созревание': ['SPEI_07', 'SPEI_08', 'SPEI_09'],  
}

for stage_name, cols in growth_stages.items():
    spei_wide[f'SPEI_{stage_name}'] = spei_wide[cols].mean(axis=1)
    print(f"  OK {stage_name}: {cols} -> SPEI_{stage_name}")

# ========== 4. Обработка данных урожайности ==========
yield_df['year'] = yield_df['year'].astype(int)
# Перевод из центнеров на гектар (ц/га) в тонны на гектар (т/га)
yield_df['yield_per_hectare'] = yield_df['yield_per_hectare'] * 10.0

yield_col = 'yield_per_hectare' if 'yield_per_hectare' in yield_df.columns else yield_df.columns[1]
print(f"\nСтолбец урожайности: {yield_col}")

yield_df = yield_df.reset_index(drop=True)
window = 5
yield_df['trend_yield'] = yield_df[yield_col].rolling(window=window, center=True).mean()

for i in range(len(yield_df)):
    if pd.isna(yield_df.loc[i, 'trend_yield']):
        start = max(0, i-2)
        end = min(len(yield_df), i+3)
        available = yield_df.loc[start:end, yield_col].dropna()
        if len(available) >= 3:
            yield_df.loc[i, 'trend_yield'] = available.mean()
yield_df['trend_yield'] = yield_df['trend_yield'].interpolate()

yield_df['weather_yield'] = yield_df[yield_col] - yield_df['trend_yield']
yield_df['relative_weather_yield'] = (yield_df['weather_yield'] / yield_df['trend_yield']) * 100

# ========== 5. Объединение данных ==========
merged = yield_df.merge(spei_wide, on='year', how='inner')
print(f"Объединённые данные: {len(merged)} лет")
print(f"Количество столбцов: {len(merged.columns)}")

# ========== 6. Корреляционный анализ ==========
print("\n" + "="*60)
print("Корреляция SPEI по фазам с метеорологической урожайностью")
print("="*60)
print(f"{'Фаза':<20} {'Абсолютная метеоурожайность':>25} {'Относительная метеоурожайность':>25}")
print(f"{'':<20} {'r':>8} {'p':>8} {'Знач.':<6} {'r':>8} {'p':>8} {'Знач.':<6}")
print("-"*60)

results = []
for stage_name in growth_stages.keys():
    spei_values = merged[f'SPEI_{stage_name}']
    
    r_abs, p_abs = stats.pearsonr(spei_values, merged['weather_yield'])
    sig_abs = '***' if p_abs < 0.01 else ('**' if p_abs < 0.05 else ('*' if p_abs < 0.1 else 'ns'))
    
    r_rel, p_rel = stats.pearsonr(spei_values, merged['relative_weather_yield'])
    sig_rel = '***' if p_rel < 0.01 else ('**' if p_rel < 0.05 else ('*' if p_rel < 0.1 else 'ns'))
    
    print(f"{stage_name:<20} {r_abs:>8.4f} {p_abs:>8.4f} {sig_abs:<6} {r_rel:>8.4f} {p_rel:>8.4f} {sig_rel:<6}")
    
    results.append({
        'stage': stage_name,
        'r_abs': r_abs, 'p_abs': p_abs, 'sig_abs': sig_abs,
        'r_rel': r_rel, 'p_rel': p_rel, 'sig_rel': sig_rel
    })

print("-"*60)
print("Обозначения: *** p<0.01, ** p<0.05, * p<0.1, ns незначимо")

# ========== 7. Визуализация ==========

# ---------- Рисунок: Диаграммы рассеяния по четырём ключевым фазам ----------
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))

key_stages = ['Весь сезон', 'Всходы-вытягивание', 'Вытягивание-цветение', 'Цветение-созревание']
plot_titles = ['(a) Весь сезон (V–IX)', '(b) Всходы–вытягивание (V–VI)',
               '(c) Вытягивание–цветение (VI–VII)', '(d) Цветение–созревание (VII–IX)']

for idx, (stage_name, title) in enumerate(zip(key_stages, plot_titles)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    spei_values = merged[f'SPEI_{stage_name}']
    rel_yield = merged['relative_weather_yield']
    
    stage_result = next(r for r in results if r['stage'] == stage_name)
    
    # Диаграмма рассеяния
    sc = ax.scatter(spei_values, rel_yield, c=merged['year'], 
                    cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
    
    # Линия тренда
    z = np.polyfit(spei_values, rel_yield, 1)
    p_trend = np.poly1d(z)
    x_line = np.linspace(-1.6, 1.6, 100)
    ax.plot(x_line, p_trend(x_line), 'r-', linewidth=2, 
            label=f'r = {stage_result["r_rel"]:.3f}, p = {stage_result["p_rel"]:.3f}')
    
    # Нулевые линии
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Выделение засушливых лет с недобором урожая
    drought_loss = merged[(spei_values < -0.5) & (rel_yield < -5)]
    if len(drought_loss) > 0:
        ax.scatter(drought_loss[f'SPEI_{stage_name}'], drought_loss['relative_weather_yield'], 
                  s=100, facecolors='none', edgecolors='red', linewidth=2, 
                  label=f'Засуха + недобор (n={len(drought_loss)})')
    
    # Подписи осей
    ax.set_xlabel(f'SPEI {stage_name}'.replace('-', '–'), fontsize=10)
    ax.set_ylabel('Относительная метеорологическая\nурожайность (%)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    if stage_name == 'Весь сезон':
        ax.set_ylim(-28, 28)

# Унификация диапазона оси X для всех подграфиков
for ax in axes.flat:
    ax.set_xlim(-1.6, 1.6)

# Общий заголовок
plt.suptitle('Влияние засухи на урожайность кукурузы по фазам развития\nв провинции Цзилинь (1980–2020)', 
             fontsize=14, fontweight='bold', y=0.99)
plt.subplots_adjust(top=0.90, hspace=0.45, wspace=0.40)
plt.savefig('growth_stage_scatter.png', dpi=300, bbox_inches='tight')
print("Диаграммы рассеяния сохранены: growth_stage_scatter.png")
plt.show()

# ========== 8. Поиск наиболее чувствительной фазы ==========
print("\n" + "="*60)
print("Ключевые выводы")
print("="*60)

best_abs = max(results, key=lambda x: abs(x['r_abs']))
best_rel = max(results, key=lambda x: abs(x['r_rel']))

print(f"\n--- Фаза с наибольшей корреляцией (абс. метео-урожайность):")
print(f"   {best_abs['stage']}: r = {best_abs['r_abs']:.4f}, p = {best_abs['p_abs']:.4f} ({best_abs['sig_abs']})")

print(f"\n--- Фаза с наибольшей корреляцией (отн. метео-урожайность):")
print(f"   {best_rel['stage']}: r = {best_rel['r_rel']:.4f}, p = {best_rel['p_rel']:.4f} ({best_rel['sig_rel']})")

significant_stages = [r for r in results if r['p_rel'] < 0.1 or r['p_abs'] < 0.1]
if significant_stages:
    print(f"\n--- Значимые или близкие к значимым фазы (p<0.1):")
    for s in significant_stages:
        print(f"   {s['stage']}: отн. r={s['r_rel']:.3f} (p={s['p_rel']:.3f}), абс. r={s['r_abs']:.3f} (p={s['p_abs']:.3f})")
else:
    print(f"\n--- Значимых фаз не обнаружено (p>0.1)")

# Помесячный анализ
print(f"\n--- Помесячная корреляция SPEI с отн. метео-урожайностью:")
for month in [5, 6, 7, 8, 9]:
    col_name = f'SPEI_{month:02d}'
    r_month, p_month = stats.pearsonr(merged[col_name], merged['relative_weather_yield'])
    sig = '***' if p_month < 0.01 else ('**' if p_month < 0.05 else ('*' if p_month < 0.1 else 'ns'))
    bar = '|' * int(abs(r_month) * 30)
    print(f"  Месяц {month} (SPEI_{month:02d}): r = {r_month:+.4f}, p = {p_month:.4f} {sig} {bar}")

# ========== 9. Сохранение результатов ==========
merged.to_csv('stage_spei_yield.csv', index=False, encoding='utf-8-sig')
print(f"\n--- Результаты сохранены в stage_spei_yield.csv")

print("\n" + "="*60)
print("Таблица результатов для отчёта")
print("="*60)
result_df = pd.DataFrame(results)
result_df = result_df.round(4)
print(result_df.to_string(index=False))