import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np

# --- Загрузка пространственных данных ---
working_dir = os.getcwd()
boundary_file = os.path.join(working_dir, "Граница провинции Цзилинь.geojson")

if not os.path.exists(boundary_file):
    raise FileNotFoundError(f"Файл границы не найден: {boundary_file}")

# Чтение границы провинции Цзилинь (WGS84)
jilin_geometry = gpd.read_file(boundary_file)
if jilin_geometry.crs is None:
    jilin_geometry = jilin_geometry.set_crs("EPSG:4326")
elif jilin_geometry.crs.to_string() != "EPSG:4326":
    jilin_geometry = jilin_geometry.to_crs("EPSG:4326")

# --- Обработка растров SPEI ---
target_tifs = sorted([f for f in os.listdir(working_dir) if f.endswith('.tif') and f.startswith('SPEI')])
print(f"Найдено файлов: {len(target_tifs)}")

results = []

for tif_name in target_tifs:
    # Извлечение года и месяца из имени файла (формат: SPEI_3_2011_1.tif)
    parts = tif_name.replace('.tif', '').split('_')
    try:
        year = int(parts[2])
        month = int(parts[3])
    except (IndexError, ValueError):
        print(f"[Пропуск] Неверный формат имени: {tif_name}")
        continue
    
    tif_path = os.path.join(working_dir, tif_name)
    
    with rasterio.open(tif_path) as src:
        try:
            # Обрезка по границе провинции и извлечение первого (единственного) канала
            out_image, _ = mask(src, jilin_geometry.geometry, crop=True)
            out_image = out_image[0]  # Преобразование из 3D (1, height, width) в 2D (height, width)
            
            # Определение значения NoData
            nodata = src.nodata if src.nodata is not None else -32768
            
            # Фильтрация: удаление NoData, NaN и выбросов
            valid = out_image[(out_image != nodata) & (~np.isnan(out_image)) & (out_image > -9990)]
            
            if valid.size == 0:
                spei_mean = np.nan
            else:
                # Масштабный коэффициент 0.01 для восстановления фактических значений SPEI
                spei_mean = float(valid.mean()) * 0.01
            
            results.append({'year': year, 'month': month, 'SPEI': spei_mean})
            print(f"Обработано: {tif_name} | SPEI = {spei_mean:.4f}")
            
        except Exception as e:
            print(f"[Ошибка] Обрезка {tif_name}: {e}")

# --- Сохранение результата ---
df = pd.DataFrame(results)
df = df.sort_values(['year', 'month'])

output_path = os.path.join(working_dir, 'jilin_spei_monthly.csv')
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nГотово! Сохранено: {output_path}")
print(f"Всего месяцев: {len(results)}")
print(df.head(10))