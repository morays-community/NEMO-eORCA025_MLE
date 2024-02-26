# modules

# utils
import os
import sys
import re
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

to_plot = dict()
infos = dict()
# ============================================================
#                         Plots Infos
# ============================================================
infos[ 'sosstsst' ] = [ 'SST (degC)', cmocean.cm.thermal , colors.Normalize(vmin=-2, vmax=35), lambda x: x ]
infos[ 'sosaline' ] = [ 'Salinity (psu)' , cmocean.cm.haline , colors.Normalize(vmin=28, vmax=38), lambda x: x ]
infos[ 'sossheig' ] = [ 'SSH (m)' , cmocean.cm.deep , colors.Normalize(vmin=-2, vmax=2), lambda x: x ]
infos[ 'soextrho' ] = [ '-\u0394\u03c1 (kg/m³)' , cmocean.cm.ice_r , colors.LogNorm(vmin=0.000005, vmax=0.05), lambda x: abs(x) ]
infos[ 'vozocrtx' ] = [ 'u (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-1.0, vmax=1.0), lambda x: x ]
infos[ 'vomecrty' ] = [ 'v (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.5, vmax=0.5), lambda x: x ]
infos[ 'vovecrtz' ] = [ 'w (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.0000005, vmax=0.0000005), lambda x: x]
infos[ 'sohefldo' ] = [ 'Heat Flux (W/m2)' , cmocean.cm.balance , colors.Normalize(vmin=-150., vmax=150.), lambda x: x ]
infos[ 'sosfldow' ] = [ 'Salt Flux (g/m2/s)' , cmocean.cm.amp , colors.Normalize(vmin=0.0, vmax=0.002), lambda x: x ]
infos[ 'sowaflup' ] = [ 'Fresh Water Flux (kg/m2/s)' , cmocean.cm.tarn , colors.Normalize(vmin=-0.0005, vmax=0.0005), lambda x: x ]
infos[ 'voeke' ] = [ 'EKE (cm2/s2)' , cmocean.cm.amp , colors.LogNorm(vmin=1., vmax=5000.), lambda x: x*1000. ]
infos[ 'vomke' ] = [ 'MKE (cm2/s2)' , cmocean.cm.amp , colors.LogNorm(vmin=1., vmax=5000.), lambda x: x*1000. ]
infos[ 'votke' ] = [ 'TKE (cm2/s2)' , cmocean.cm.amp , colors.LogNorm(vmin=1., vmax=5000.), lambda x: x*1000. ]
infos[ 'vomevt' ] = [ 'T.v (degC.m/s)' , cmocean.cm.balance , colors.Normalize(vmin=-18.0, vmax=18.0), lambda x: x ]
infos[ 'vomevs' ] = [ 'S.v (psu.m/s)' , cmocean.cm.delta , colors.Normalize(vmin=-18.0, vmax=18.0), lambda x: x ]
infos[ 'vozout' ] = [ 'T.u (degC.m/s)' , cmocean.cm.balance , colors.Normalize(vmin=-35.0, vmax=35.0), lambda x: x ]
infos[ 'vozous' ] = [ 'S.u (degC.m/s)' , cmocean.cm.delta , colors.Normalize(vmin=-35.0, vmax=35.0), lambda x: x ]
infos[ 'votemper' ] = [ 'Temperature (degC)' , cmocean.cm.thermal , colors.Normalize(vmin=-2, vmax=35), lambda x: x ]
infos[ 'vosaline' ] = [ 'Salinity (psu)' , cmocean.cm.haline , colors.Normalize(vmin=28, vmax=38), lambda x: x ]
infos[ 'voextrho' ] = [ '-\u0394\u03c1 (kg/m³)' , cmocean.cm.ice_r , colors.LogNorm(vmin=0.000005, vmax=0.05), lambda x: x ]
# ============================================================
#                           2D Fields
# ============================================================
to_plot[ 'gridTsurf' ] = ['soextrho', 'sosstsst','sosaline','soextrho','sossheig'] # sst, sss, drho
to_plot[ 'gridUsurf' ] = ['vozocrtx']                       # u-current
to_plot[ 'gridVsurf' ] = ['vomecrty']                       # v-current
to_plot[ 'flxT' ] = ['sohefldo','sosfldow','sowaflup']      # heat, salt, water fluxes
# ============================================================
#                           3D Fields
# ============================================================
to_plot[ 'gridT' ] = ['votemper','vosaline','voextrho']     # T, S, drho
to_plot[ 'gridU' ] = ['vozocrtx']                           # u-current
to_plot[ 'gridV' ] = ['vomecrty']                           # v-current
to_plot[ 'gridW' ] = ['vovecrtz']                           # w-current
to_plot[ 'EKE' ] = ['voeke','vomke','votke']                # eke, mke, tke
to_plot[ 'VT' ]  = ['vomevt','vomevs','vozout','vozous']    # vt, vs, ut, us
# ============================================================


def main(respath,plotpath,confcase,freq,years):
    global to_plot
    global infos

    # loop on xios freq directories
    for freq in freqs:
        print(f'\n====== {freq} ======')
        freq_path = respath+'/'+freq
        if not os.path.exists(freq_path) and not os.path.isdir(freq_path):
            print(f'ERROR: {freq_path} does not exist')
            sys.exit()
        os.chdir(freq_path)
        if years[0] is None:
            years = os.listdir(os.getcwd())        

        # loop on years directories
        for i,year in enumerate(years):
            print(f'\n{year}:')
            print(f'----')
            year_path = freq_path+'/'+year
            file_base = year_path + '/'+confcase+'_y'+year+'.'+freq
            os.chdir(year_path)

            # check output directory
            outpath = plotpath + '/' + freq + '/' + year
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # loop on yearly files
            for file in to_plot.keys():
                grid_file = file_base + '_' + file + '.nc'
                
                # get mesh and data if file exists
                if os.path.isfile(grid_file):
                    ds=xr.open_dataset(grid_file)
                    lon = ds.nav_lon.values
                    lat = ds.nav_lat.values
                    try:
                        dpt = ds.deptht.values
                    except Exception as e0:
                        try:
                            dpt = ds.depthu.values
                        except Exception as e1:
                            try:
                                dpt = ds.depthv.values
                            except Exception as e2:
                                try:
                                    dpt = ds.depthw.values
                                except Exception as e3:
                                    dpt = None                    

                    # loop on fields
                    for fld in to_plot[file]:
                        print(f'Plotting {fld}')
                        data = getattr(ds,fld).values[-1]
                        # check data
                        if data is None:
                            data = np.zeros(lon.shape())
                        if dpt is not None:
                            print(f'Depth not handled for 3D fields yet, plotting only surface')
                            data = data[0,:,:]
                            
                        # plot
                        output = outpath + '/' + confcase + '_y' + year + '.' + freq + '_' + file + '_' + fld + '.png'
                        make_plot(data,lon,lat,infos[fld],output)
                else:
                    print(f'File {grid_file} not found, ignored')


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
    parser.add_argument('-c', dest='confcase', type=str, default='noconf')
    parser.add_argument('-f', nargs='+', dest='freqs', type=str, default=['1d'])
    parser.add_argument('-y', nargs='+', dest='years', type=str, default=[None])
    args = parser.parse_args()
    confcase = args.confcase
    freqs = args.freqs
    years = args.years

    # check freqs args
    for freq in freqs:
        if not re.match(r'\d+[a-zA-Z]$', freq):
            print(f'ERROR: Invalid format for "{freq}". Use format "number + letter".')
            sys.exit()

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

    # get storage dir
    if 'SDIR' in os.environ:
        SDIR = os.environ['SDIR']
    else:
        print('ERROR: Environment variable $SDIR not defined')
        sys.exit()

    # path to MEAN
    print(f'Storage dir: {SDIR}')
    print(f'CONFCASE : {confcase}')
    conf, case = confcase.split('-')
    respath = SDIR + '/' + conf + '/' + confcase + '-MEAN'
    if not os.path.exists(respath) and not os.path.isdir(respath):
        print(f'ERROR: {respath} does not exist')
        sys.exit()

    # path to PLOTS
    plotpath = SDIR + '/' + conf + '/' + confcase + '-PLOTS'
    print(f'Plot dir: {plotpath}')
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)

    # proceed
    main(respath,plotpath,confcase,freqs,years)
