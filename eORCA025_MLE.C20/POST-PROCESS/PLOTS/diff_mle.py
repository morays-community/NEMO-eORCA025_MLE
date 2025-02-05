import os
import argparse
import numpy as np
import xarray as xr
import cmocean

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors


def main(filepath,filepath_ref,plotpath):

    print(filepath)
    print(filepath_ref)

    # read files
    ds = xr.open_dataset(filepath)
    ds_ref = xr.open_dataset(filepath_ref)
    lon = ds_ref.nav_lon.values
    lat = ds_ref.nav_lat.values

    # get fields
    mld = getattr(ds,"somxl010").values[-1] 
    mld_ref = getattr(ds_ref,"somxl010").values[-1]
    diff_mld = mld - mld_ref

    # plot
    lim = 400.
    infos = [ 'MLD-mle - MLD (m)' , cmocean.cm.balance , colors.Normalize(vmin=-lim, vmax=lim), lambda x: x ]
    make_plot(diff_mld,lon,lat,infos,plotpath)


def make_plot(data,lon,lat,infos,output):
    # args
    title, cmap, norm, tfs = infos
    data = tfs(data)
    # figure
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    # colo map
    pcm = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
    plt.title(title)
    # write fig
    plt.savefig(output, bbox_inches='tight')
    plt.close()


if __name__=="__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-cref', dest='confcase_ref', type=str, default='noconf')
    parser.add_argument('-c', dest='confcase', type=str, default='noconf')
    parser.add_argument('-f', dest='freq', type=str, default='1d')
    parser.add_argument('-y', dest='year', type=str, default=None)
    parser.add_argument('-m', dest='month', type=str, default=None)
    args = parser.parse_args()
    confcase_ref = args.confcase_ref
    confcase = args.confcase
    freq = args.freq
    year = args.year
    month = "{:02d}".format(int(args.month))

    # guess CONFCASE if not given
    if confcase == 'noconf':
        cur_dir = os.getcwd()
        cur_dir = cur_dir.split(os.path.sep)
        
        if cur_dir[-1] != 'CDF' and cur_dir[-2] != 'CTL':
            # we are not in CDF/CTL dir and confcase not given, impossible to guess
            print('ERROR: must be executed in the CONFCASE CTL directory')
            raise SystemExit
        else:
            confcase = cur_dir[-3]

    # path to MEAN
    SDIR = os.environ['SDIR']
    print(f'Storage dir: {SDIR}')
    print(f'CONFCASE : {confcase}')
    conf, case = confcase.split('-')
    respath = SDIR + '/' + conf + '/' + confcase + '-MEAN'
    respath_ref = SDIR + '/' + conf + '/' + confcase_ref + '-MEAN'
    if not os.path.exists(respath) and not os.path.isdir(respath):
        print(f'ERROR: {respath} does not exist')
        sys.exit()

    # get files path
    filepath = respath + "/" + freq + "/" + year + "/" + confcase + "_y" + year + "m" + month + "." + freq + "_gridTsurf.nc"
    filepath_ref = respath_ref + "/" + freq + "/" + year + "/" + confcase_ref + "_y" + year + "m" + month + "." + freq + "_gridTsurf.nc"
    plotpath = SDIR + '/' + conf + '/' + confcase + '-PLOTS/' + freq + "/" + year + "/" + confcase + "_y" + year + "m" + month + "." + freq + "_diffmld.png"

    main(filepath,filepath_ref,plotpath)
