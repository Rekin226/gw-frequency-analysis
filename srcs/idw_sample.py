import geopandas as gpd
import pandas as pd
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from pyproj import CRS, Transformer
from matplotlib.lines import Line2D

# 設定字體和負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定義常數參數
twd97_crs = CRS.from_string("+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +units=m +no_defs")
wgs84_crs = CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")
transformer = Transformer.from_crs(twd97_crs, wgs84_crs)
#shp_file = '../to_output_the_data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp'batch用
shp_file = '../data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp'
grid_size = 200  # 網格大小，用於生成插值點

# 讀取GIS.csv文件
#df = pd.read_csv('../workspace/Results/(1)gis_model-55.csv', encoding='big5') batch用
df = pd.read_csv('../workspace/Results/gis_model.csv', encoding='big5')
# 讀取濁水溪範圍的shp檔案
boundary = gpd.read_file(shp_file)
boundary_bounds = boundary.total_bounds
boundary = boundary.to_crs('EPSG:4326')
boundary_bounds = boundary.total_bounds

# 提取X和Y座標以及值
x = df['groundwater_TM_X97']
y = df['groundwater_TM_Y97']
x, y = transformer.transform(list(x), list(y))

# 定義要繪製的數據
data = {
    'values1': ['a1', 'b1', 'z1'],
    #'values2': ['a2', 'b2', 'z2'],
    'values3': ['a3', 'b3', 'z3'],
}

# 繪製圖形
fig, ax = plt.subplots(len(data), len(data['values1']), figsize=(18, 6*len(data)))
#fig.suptitle('IDW Interpolation')

for row, (key, values) in enumerate(data.items()):
    for col, value in enumerate(values):
        axis = ax[row, col]

        # 使用Rbf進行IDW內插
        rbf = Rbf(x, y, df[value], function='inverse')

        # 定義插值範圍
        x_min, x_max = boundary_bounds[0], boundary_bounds[2]
        y_min, y_max = boundary_bounds[1], boundary_bounds[3]
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # 計算插值
        zi = rbf(xi, yi)

        # 散點圖和標籤
        axis.scatter(x, y, marker='.')
        for j, name in enumerate(df['groundwater_name']):
            axis.annotate(name, (x[j], y[j]), fontsize=7)

        # 繪製等高線圖
        num_levels = 20
        axis.set_title(value)
        axis.ticklabel_format(useOffset=False, style='plain')
        axis.set_xlim(boundary_bounds[0], boundary_bounds[2])
        axis.set_ylim(boundary_bounds[1], boundary_bounds[3])
        boundary.plot(ax=axis, facecolor='none', edgecolor='black')
        contour = axis.contourf(xi, yi, zi, levels=num_levels, cmap='coolwarm', alpha=0.7)

        # 創建濁水溪範圍的多邊形
        boundary_polygon = mpath.Path(boundary.geometry.values[0].exterior.coords)
        clip_path = PathPatch(boundary_polygon, transform=axis.transData)

        # 對等高線圖進行遮罩處理
        for collection in contour.collections:
            collection.set_clip_path(clip_path)

        # 添加色標
        cbar = fig.colorbar(contour, ax=axis, label='Value')
        axis.grid(True)

        # 添加濁水溪範圍的黑框到圖例中
        legend_elements = [
            Line2D([], [], color='black', linewidth=1, label='AREA'),
            Line2D([], [], marker='o', color='b', label='ST', linestyle='None')]
        axis.add_artist(axis.legend(handles=legend_elements, loc='upper left', fontsize='small'))

# 調整子圖間距並顯示圖形
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()
