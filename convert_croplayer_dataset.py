"""
convert_croplayer_dataset.py

将CropLayer shp矢量标注 + Mapbox卫星影像 转换为YOLO实例分割数据集。

前置条件:
1. 在 https://account.mapbox.com/ 注册免费账号，获取access token
2. 已下载并解压CropLayer北京数据到 /home/fumu/datadisk/croplayer_beijing/

使用方法:
    python convert_croplayer_dataset.py --token YOUR_MAPBOX_TOKEN

可选参数:
    --token       Mapbox access token (必填, 或设置MAPBOX_TOKEN环境变量)
    --shp-dir     shp文件目录 (默认: /home/fumu/datadisk/croplayer_beijing/cf_county_beijing)
    --zoom        Mapbox瓦片缩放级别 (默认: 15, 约1.83m/pixel@2x)
    --retina      使用@2x高清瓦片 (默认: True)
    --tile-size   YOLO子图尺寸 (默认: 640)
    --step        子图网格步长 (默认: 等于tile-size, 无重叠)
    --min-area    最小多边形像素面积 (默认: 200)
    --min-parcels 每区最少地块数 (默认: 50, 低于此数的城区跳过)
    --val-ratio   验证集比例 (默认: 0.2)
    --output-dir  输出目录 (默认: datasets/croplayer)
    --cache-dir   瓦片缓存目录 (默认: /home/fumu/datadisk/croplayer_beijing/tiles)
    --seed        随机种子 (默认: 42)
"""

import os
import sys
import math
import argparse
import time
import numpy as np
from PIL import Image
import requests
from pathlib import Path
from io import BytesIO
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon
import yaml
import random


SEED = 42


def lat_lon_to_tile_frac(lat, lon, zoom):
    n = 2.0 ** zoom
    x_frac = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_frac = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x_frac, y_frac


def lat_lon_to_tile(lat, lon, zoom):
    x_frac, y_frac = lat_lon_to_tile_frac(lat, lon, zoom)
    return int(math.floor(x_frac)), int(math.floor(y_frac))


def tile_to_lat_lon(tx, ty, zoom):
    n = 2.0 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lat, lon


def resolution_m_per_px(zoom, lat, retina=True):
    m = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)
    if retina:
        m /= 2.0
    return m


def download_mapbox_tile(z, x, y, token, retina=True, retries=3, delay=0.5):
    suffix = "@2x" if retina else ""
    url = f"https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}{suffix}.png?access_token={token}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert('RGB')
            elif resp.status_code == 404:
                return None
            else:
                print(f"  tile {z}/{x}/{y} status {resp.status_code}, retry {attempt+1}")
                time.sleep(delay)
        except Exception as e:
            time.sleep(delay * (attempt + 1))
    return None


def geo_to_mosaic_px(lon, lat, x_min_tile, y_min_tile, zoom, tile_px_size):
    x_frac, y_frac = lat_lon_to_tile_frac(lat, lon, zoom)
    px = (x_frac - x_min_tile) * tile_px_size
    py = (y_frac - y_min_tile) * tile_px_size
    return px, py


def assemble_yolo_tile(start_px, start_py, size, cache_dir, x_min_tile, y_min_tile, tile_px_size):
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    tx_start = x_min_tile + start_px // tile_px_size
    tx_end = x_min_tile + (start_px + size - 1) // tile_px_size
    ty_start = y_min_tile + start_py // tile_px_size
    ty_end = y_min_tile + (start_py + size - 1) // tile_px_size

    for ty in range(ty_start, ty_end + 1):
        for tx in range(tx_start, tx_end + 1):
            tile_path = cache_dir / f'{tx}_{ty}.png'
            if not tile_path.exists():
                continue
            tile_img = Image.open(tile_path).convert('RGB')
            tile_mosaic_x = (tx - x_min_tile) * tile_px_size
            tile_mosaic_y = (ty - y_min_tile) * tile_px_size

            src_x0 = max(0, start_px - tile_mosaic_x)
            src_y0 = max(0, start_py - tile_mosaic_y)
            src_x1 = min(tile_px_size, start_px + size - tile_mosaic_x)
            src_y1 = min(tile_px_size, start_py + size - tile_mosaic_y)

            dst_x0 = max(0, tile_mosaic_x - start_px)
            dst_y0 = max(0, tile_mosaic_y - start_py)

            if src_x1 <= src_x0 or src_y1 <= src_y0:
                continue

            crop = tile_img.crop((src_x0, src_y0, src_x1, src_y1))
            canvas.paste(crop, (dst_x0, dst_y0))

    return canvas


def main():
    parser = argparse.ArgumentParser(description='Convert CropLayer to YOLO instance segmentation dataset')
    parser.add_argument('--token', type=str, default=os.environ.get('MAPBOX_TOKEN', ''))
    parser.add_argument('--shp-dir', type=str,
                        default='/home/fumu/datadisk/croplayer_beijing/cf_county_beijing')
    parser.add_argument('--zoom', type=int, default=15)
    parser.add_argument('--retina', action='store_true', default=True)
    parser.add_argument('--no-retina', dest='retina', action='store_false')
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--min-area', type=int, default=200)
    parser.add_argument('--min-parcels', type=int, default=50)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='datasets/croplayer')
    parser.add_argument('--cache-dir', type=str,
                        default='/home/fumu/datadisk/croplayer_beijing/tiles')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not args.token:
        print("ERROR: 需要Mapbox access token。使用 --token 参数或设置 MAPBOX_TOKEN 环境变量。")
        print("注册地址: https://account.mapbox.com/")
        sys.exit(1)

    zoom = args.zoom
    retina = args.retina
    yolo_size = args.tile_size
    step = args.step or yolo_size
    min_area = args.min_area
    min_parcels = args.min_parcels
    val_ratio = args.val_ratio
    seed = args.seed
    tile_px_size = 512 if retina else 256

    random.seed(seed)

    # ===== Step 1: 加载shp标注 =====
    print("=" * 60)
    print("Step 1: 加载shp矢量标注")
    print("=" * 60)

    shp_dir = Path(args.shp_dir)
    shp_files = sorted(shp_dir.glob('*.shp'))
    if not shp_files:
        print(f"ERROR: 未找到shp文件 in {shp_dir}")
        sys.exit(1)

    gdfs = []
    for shp_file in shp_files:
        gdf = gpd.read_file(shp_file)
        district = shp_file.stem.split('_')[-1]
        n = len(gdf)
        if n < min_parcels:
            print(f"  跳过 {district}: {n}个地块 (低于阈值{min_parcels})")
            continue
        print(f"  包含 {district}: {n}个地块")
        gdfs.append(gdf)

    if not gdfs:
        print("ERROR: 没有满足条件的区!")
        sys.exit(1)

    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
    n_parcels_total = len(gdf_all)
    print(f"合计: {n_parcels_total}个地块")

    b = gdf_all.geometry.bounds
    lon_min = float(b.minx.min())
    lat_min = float(b.miny.min())
    lon_max = float(b.maxx.max())
    lat_max = float(b.maxy.max())

    lon_margin = (lon_max - lon_min) * 0.02
    lat_margin = (lat_max - lat_min) * 0.02
    lon_min -= lon_margin
    lon_max += lon_margin
    lat_min -= lat_margin
    lat_max += lat_margin

    center_lat = (lat_min + lat_max) / 2
    m_per_px = resolution_m_per_px(zoom, center_lat, retina)
    coverage_m = yolo_size * m_per_px
    print(f"地理范围: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"分辨率: ~{m_per_px:.2f} m/pixel @ lat={center_lat:.2f}°")
    print(f"YOLO子图地面覆盖: {coverage_m:.0f}m × {coverage_m:.0f}m ({coverage_m*coverage_m/1e6:.2f} km²)")

    # 计算瓦片范围
    tx_nw, ty_nw = lat_lon_to_tile(lat_max, lon_min, zoom)
    tx_se, ty_se = lat_lon_to_tile(lat_min, lon_max, zoom)

    x_min_tile = tx_nw - 2
    y_min_tile = ty_nw - 2
    x_max_tile = tx_se + 2
    y_max_tile = ty_se + 2

    n_tiles_x = x_max_tile - x_min_tile + 1
    n_tiles_y = y_max_tile - y_min_tile + 1
    total_mapbox_tiles = n_tiles_x * n_tiles_y
    mosaic_w = n_tiles_x * tile_px_size
    mosaic_h = n_tiles_y * tile_px_size

    print(f"瓦片范围: x [{x_min_tile}, {x_max_tile}], y [{y_min_tile}, {y_max_tile}]")
    print(f"需下载: {n_tiles_x} × {n_tiles_y} = {total_mapbox_tiles} 个瓦片")
    print(f"虚拟镶嵌图: {mosaic_w} × {mosaic_h} 像素")

    # ===== Step 2: 将polygon转为镶嵌图像素坐标 =====
    print("\n" + "=" * 60)
    print("Step 2: 转换polygon坐标为像素坐标")
    print("=" * 60)

    all_polygons_px = []
    all_centroids_px = []

    for idx, row in gdf_all.iterrows():
        geom = row.geometry
        if geom.geom_type != 'Polygon':
            continue
        coords = list(geom.exterior.coords)
        px_coords = []
        for lon_v, lat_v in coords:
            px, py = geo_to_mosaic_px(lon_v, lat_v, x_min_tile, y_min_tile, zoom, tile_px_size)
            px_coords.append((px, py))

        poly_px = Polygon(px_coords)
        if not poly_px.is_valid:
            poly_px = poly_px.buffer(0)
        if poly_px.area >= min_area:
            all_polygons_px.append(poly_px)
            all_centroids_px.append((poly_px.centroid.x, poly_px.centroid.y))

    print(f"有效polygon(像素坐标): {len(all_polygons_px)}")
    if all_polygons_px:
        areas = [p.area for p in all_polygons_px]
        print(f"像素面积: min={min(areas):.1f}, median={sorted(areas)[len(areas)//2]:.1f}, max={max(areas):.1f}")

    # ===== Step 3: 下载Mapbox卫星瓦片 =====
    print("\n" + "=" * 60)
    print("Step 3: 下载Mapbox卫星瓦片")
    print("=" * 60)

    cache_subdir = Path(args.cache_dir) / f'z{zoom}' / ('retina' if retina else 'standard')
    cache_subdir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    cached = 0
    failed = 0
    start_time = time.time()

    for ty in range(y_min_tile, y_max_tile + 1):
        for tx in range(x_min_tile, x_max_tile + 1):
            cache_path = cache_subdir / f'{tx}_{ty}.png'

            if cache_path.exists():
                cached += 1
                continue

            img = download_mapbox_tile(zoom, tx, ty, args.token, retina)
            if img is not None:
                img.save(cache_path)
                downloaded += 1
            else:
                placeholder = Image.new('RGB', (tile_px_size, tile_px_size), (0, 0, 0))
                placeholder.save(cache_path)
                failed += 1

            total_processed = downloaded + failed + cached
            if total_processed % 100 == 0 and total_processed > 0:
                elapsed = time.time() - start_time
                remaining = total_mapbox_tiles - total_processed
                rate = (downloaded + failed) / elapsed if elapsed > 0 else 0
                eta_min = remaining / rate / 60 if rate > 0 else 999
                print(f"  进度: {downloaded}下载, {failed}失败, {cached}缓存, ETA {eta_min:.1f}min")

            time.sleep(0.1)

    elapsed = time.time() - start_time
    print(f"  完成: {downloaded}下载, {failed}失败, {cached}缓存, 耗时{elapsed/60:.1f}min")

    # ===== Step 4: 生成YOLO数据集 =====
    print("\n" + "=" * 60)
    print("Step 4: 生成YOLO数据集")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    img_train_dir = output_dir / 'images' / 'train'
    img_val_dir = output_dir / 'images' / 'val'
    lbl_train_dir = output_dir / 'labels' / 'train'
    lbl_val_dir = output_dir / 'labels' / 'val'

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 将每个polygon分配到包含其centroid的YOLO子图
    tile_assignments = {}
    for pidx, (cx, cy) in enumerate(all_centroids_px):
        gx = int(cx // step) * step
        gy = int(cy // step) * step
        gx = max(0, min(gx, mosaic_w - yolo_size))
        gy = max(0, min(gy, mosaic_h - yolo_size))
        key = (gx, gy)
        if key not in tile_assignments:
            tile_assignments[key] = []
        tile_assignments[key].append(pidx)

    # 随机划分train/val
    tile_keys = sorted(tile_assignments.keys())
    random.shuffle(tile_keys)
    n_val = int(len(tile_keys) * val_ratio)
    val_keys = set(tile_keys[:n_val])

    print(f"有标注的子图数: {len(tile_keys)}")
    print(f"Train: {len(tile_keys) - n_val}, Val: {n_val}")

    tile_box = box(0, 0, yolo_size, yolo_size)
    img_count = 0
    label_stats = {'total_labels': 0, 'total_images': 0, 'avg_labels': 0}

    for key in sorted(tile_assignments.keys()):
        gx, gy = key
        is_val = key in val_keys
        split = 'val' if is_val else 'train'

        img = assemble_yolo_tile(gx, gy, yolo_size, cache_subdir, x_min_tile, y_min_tile, tile_px_size)

        arr = np.array(img)
        if np.mean(arr) < 10:
            continue

        poly_indices = tile_assignments[key]
        label_lines = []

        for pidx in poly_indices:
            poly = all_polygons_px[pidx]
            coords = [(x - gx, y - gy) for x, y in poly.exterior.coords]
            poly_tile = Polygon(coords)

            if not poly_tile.is_valid:
                poly_tile = poly_tile.buffer(0)

            clipped = poly_tile.intersection(tile_box)
            if clipped.is_empty:
                continue

            polys_to_write = []
            if clipped.geom_type == 'Polygon':
                polys_to_write.append(clipped)
            elif clipped.geom_type == 'MultiPolygon':
                for sub in clipped.geoms:
                    if sub.geom_type == 'Polygon' and not sub.is_empty:
                        polys_to_write.append(sub)
            else:
                continue

            for sub_poly in polys_to_write:
                area_px = sub_poly.area
                if area_px < min_area:
                    continue

                coords_out = list(sub_poly.exterior.coords)
                if len(coords_out) < 4:
                    continue
                if coords_out[0] == coords_out[-1]:
                    coords_out = coords_out[:-1]
                if len(coords_out) < 3:
                    continue

                norm_coords = []
                for x_v, y_v in coords_out:
                    nx = max(0.0, min(1.0, x_v / yolo_size))
                    ny = max(0.0, min(1.0, y_v / yolo_size))
                    norm_coords.append(f'{nx:.6f} {ny:.6f}')

                label_lines.append('0 ' + ' '.join(norm_coords))

        if not label_lines:
            continue

        fname = f'croplayer_{img_count:04d}'
        img_dir = img_val_dir if is_val else img_train_dir
        lbl_dir = lbl_val_dir if is_val else lbl_train_dir

        img.save(img_dir / f'{fname}.jpg', quality=95)
        with open(lbl_dir / f'{fname}.txt', 'w') as f:
            f.write('\n'.join(label_lines))

        label_stats['total_labels'] += len(label_lines)
        label_stats['total_images'] += 1
        img_count += 1

        if img_count % 50 == 0:
            print(f"  已保存 {img_count} 张子图...")

    label_stats['avg_labels'] = label_stats['total_labels'] / label_stats['total_images'] if label_stats['total_images'] > 0 else 0

    n_train_imgs = len(list(img_train_dir.glob('*.jpg')))
    n_val_imgs = len(list(img_val_dir.glob('*.jpg')))
    print(f"\n数据集统计:")
    print(f"  Train图像: {n_train_imgs}")
    print(f"  Val图像: {n_val_imgs}")
    print(f"  总标注数: {label_stats['total_labels']}")
    print(f"  平均每图标注: {label_stats['avg_labels']:.1f}")

    # ===== Step 5: 创建data.yaml =====
    print("\n" + "=" * 60)
    print("Step 5: 创建data.yaml")
    print("=" * 60)

    data_yaml = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['cropland'],
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"data.yaml: {yaml_path}")
    print(f"\n完成! 数据集已创建在 {output_dir}")
    print(f"下一步:")
    print(f"  python train_croplayer_baseline.py")
    print(f"  python train_croplayer_yolo_star.py")


if __name__ == '__main__':
    main()