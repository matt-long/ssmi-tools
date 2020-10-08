import os
from subprocess import check_call
import re

from datetime import datetime
import cftime

import numpy as np
import xarray as xr
import struct

from glob import glob

USER = os.environ['USER']
diri = f'/glade/work/{USER}/nsidc_ssmi_seaice'
dirtmp = f'/glade/scratch/{USER}/calcs/nsidc_ssmi_seaice'

def obs_seaice(pole,res='25'):
    '''Convert binary seaice data to xr.Dataset.'''

    cached_file = f'{dirtmp}/{pole}.daily.{res}.zarr'
    if os.path.exists(cached_file):
        return xr.open_zarr(cached_file)

    bin = ssmi_binary_data(pole,res)
    ds = xr.Dataset()
    ds['lat'] = xr.DataArray(bin.lat,
                             dims=('ny','nx'),
                             name='lat',
                             attrs={'long_name':'Latitude','units':'degree_north'})
    ds['lon'] = xr.DataArray(bin.lon,
                             dims=('ny','nx'),
                             name='lon',
                             attrs={'long_name':'Longitude',
                                    'units':'degree_east'})
    ds['area'] = xr.DataArray(bin.area,
                              dims=('ny','nx'),
                              name='area',
                              attrs={'long_name':'Area',
                                     'units':'km^2'})
    ds['mask'] = xr.DataArray(bin.mask,
                              dims=('ny','nx'),
                              name='mask',
                              attrs={'long_name':'Mask',
                                     'units':''})
    if hasattr(bin,'static_pole_mask'):
        ds['static_pole_mask'] = xr.DataArray(bin.static_pole_mask,
                                              dims=('ny','nx'),
                                              name='static_pole_mask',
                                              attrs={'long_name':'Pole mask',
                                                     'units':'',
                                                     'description':'Circular mask that symmetrically covers the observed maximum extent of the missing data (resulting from the orbit inclination and instrument swath) near the North Pole. This is a 1-byte array with no header. The area of the SSM/I hole has a value of 1. The additional area of the SMMR hole (a ring around the SSM/I hole) has a value of 2. All other pixels have a value of 0.'})

    time_units = 'days since 0001-01-01'
    time = cftime.date2num(bin.time,
                           units=time_units,
                           calendar='standard')

    if hasattr(bin,'pole_mask'):
        ds['pole_mask'] = xr.DataArray(bin.pole_mask,
                                              dims=('time','ny','nx'),
                                              name='pole_mask',
                                              attrs={'long_name':'Pole mask',
                                                     'units':'',
                                                     'description':'Circular mask that symmetrically covers the observed maximum extent of the missing data (resulting from the orbit inclination and instrument swath) near the North Pole. This is a 1-byte array with no header. The area of the SSM/I hole has a value of 1. The additional area of the SMMR hole (a ring around the SSM/I hole) has a value of 2. All other pixels have a value of 0.'})


    ds['ifrac'] = xr.DataArray(bin.ice,
                               dims=('time','ny','nx'),
                               name='ifrac',
                               attrs={'long_name':'Sea ice fraction',
                                      'units':''})
    ds['time'] = xr.DataArray(time,
                              dims=('time'),
                              name='time',
                              attrs = {'long_name':'time',
                                       'units':time_units,
                                       'calendar':'standard'})

    if cached_file:
        print(f'writing {cached_file}')
        if not os.path.exists(dirtmp):
            check_call(['mkdir','-pv',dirtmp])
        ds.to_zarr(cached_file)

    return xr.decode_cf(ds)


class ssmi_binary_data(object):
    '''Interface to NSIDC binary files.'''

    def __init__(self,pole,res='25'):
        '''Constructor: reads binary files and returns numpy array as fields.'''
        self._get_files(pole,res)
        self._get_dimensions(pole,res)
        self._read_latlat_coords()
        self._read_gridarea()
        self._read_landsea_mask()
        self._read_static_pole_mask()
        self._read_icedata()

    def _get_dimensions(self,pole,res):
        '''Get the dimensions of the dataset.'''

        if res != '25':
            raise ValueError(f'Resolution not implemented: {res}')

        if pole == 'north':
            self.ny,self.nx = 448, 304
        elif pole == 'south':
            self.ny,self.nx = 332, 316
        self.nbytes = self.ny*self.nx
        self.nt = len(self.files['ice'])

    def _get_files(self,pole,res):
        '''Access the relevant files.'''

        self.files = {k:f'{diri}/polar-stereo-tools/ps{pole[0]}{res}{k}_v3.dat' for k in ['lats','lons','area']}
        self.files['mask'] = f'{diri}/polar-stereo-tools/gsfc_{res}{pole[0]}.msk'
        if pole == 'north':
            self.files['pole_mask'] = f'{diri}/polar-stereo-tools/pole_n.msk'
        self.files['ice'] = sorted(glob(f'{diri}/{pole}/*.bin'))

    def _read_latlat_coords(self):
        '''
        Grids that determine the latitude of a given pixel for the 25 km grids
        for either hemisphere (psn for the Northern Hemisphere and pss for the
        Southern Hemisphere). These latitude grids are in binary format and are
        stored as 4-byte integers (little endian) scaled by 100,000 (divide the
        stored value by 100,000 to get decimal degrees). Each array location
        (i, j) contains the latitude value at the center of the corresponding
        data grid cells.
        '''
        self.lat = self._read_coordfile(self.files['lats'],
                                        dtype='<i4',
                                        scale_factor=1e5)
        self.lon = self._read_coordfile(self.files['lons'],
                                        dtype='<i4',
                                        scale_factor=1e5)
        self.lon[self.lon<0] = self.lon[self.lon<0]+360.

    def _read_gridarea(self):
        '''
        Grids that determine the area of a given pixel for the 25 km grids for
        either hemisphere (psn for the Northern Hemisphere and pss for the
        Southern Hemisphere). The arrays are in binary format and are stored as
        4-byte integers scaled by 1000 (divide by 1000 to get square km).
        '''
        self.area = self._read_coordfile(self.files['area'],dtype='<i4',
                                         scale_factor=1e3)

    def _read_landsea_mask(self):
        '''
        25 km land and coast mask for both hemispheres
        (n: Northern Hemisphere, s: Southern Hemisphere).
        A 1-byte integer array is included in each file.
        Values are 0 or 1, where 1 is the mask.
        '''
        self.mask = self._read_coordfile(self.files['mask'],dtype='<i1',
                                         scale_factor=1.)
    def _read_static_pole_mask(self):
        '''
        Circular mask that symmetrically covers the observed maximum extent of
        the missing data (resulting from the orbit inclination and instrument
        swath) near the North Pole. This is a 1-byte array with no header.
        The area of the SSM/I hole has a value of 1. The additional area of the
        SMMR hole (a ring around the SSM/I hole) has a value of 2. All other
        pixels have a value of 0.
        '''
        if 'pole_mask' in self.files:
            self.static_pole_mask = self._read_coordfile(
                self.files['pole_mask'],dtype='<i1',scale_factor=1.)

    def _read_coordfile(self,file_in,dtype,scale_factor=1.):
        '''Read coordinate files.'''
        data = np.fromfile(file_in,dtype=dtype)/scale_factor
        data = data.reshape(self.ny,self.nx)
        data = np.flipud(data)
        return data

    def _read_icedata(self):
        '''Loop over ice files and read each.'''
        self.ice = np.empty((self.nt,self.ny,self.nx),dtype=np.float)
        self.pole_mask = np.empty((self.nt,self.ny,self.nx),dtype=np.float)
        self.time = np.empty((self.nt),dtype=datetime)

        for i,f in enumerate(self.files['ice']):
            match = re.search(r'\d{4}\d{2}\d{2}', os.path.basename(f))
            self.time[i] = datetime.strptime(match.group(), '%Y%m%d')
            self.ice[i,:,:],self.pole_mask[i,:,:] = self._read_single_icefile(f)

    def _read_single_icefile(self,file_in):
        '''Read an ice concentration file.

        Data are stored as one-byte integers representing sea ice concentration
        values. The sea ice concentration data are packed into byte format by
        multiplying the derived fractional sea ice concentration floating-point
        values (ranging from 0.0 to 1.0) by a scaling factor of 250. For
        example, a sea ice concentration value of 0.0 (0%) maps to a stored
        one-byte integer value of 0, and a sea ice concentration value of 1.0
        (100%) maps to a stored one-byte integer value of 250. To convert to the
        fractional parameter range of 0.0 to 1.0, divide the scaled data in the
        file by 250. To convert to percentage values (0% to 100%), divide the
        scaled data in the file by 2.5.

        Description of data values
        0 - 250	Sea ice concentration (fractional coverage scaled by 250)
        251	Circular mask used in the Arctic to cover the irregularly-shaped
            data gap around the pole (caused by the orbit inclination and
            instrument swath)
        252	Unused
        253	Coastlines
        254	Superimposed land mask
        255	Missing data
        '''

        with open(file_in,'rb') as fid:
            buffer = fid.read()

        data = np.array(struct.unpack_from(f'{self.nbytes}B',buffer,offset=300),
                        dtype=np.float)

        #-- apply scaling
        ice = data.copy()
        ice[data<=250] = ice[data<=250]/250.
        ice[data>250] = np.nan

        pole_mask = data.copy()
        pole_mask[:] = 0.
        pole_mask[data==251] = 1.

        #-- reshape
        ice = ice.reshape(self.ny,self.nx)
        ice = np.flipud(ice)

        pole_mask = pole_mask.reshape(self.ny,self.nx)
        pole_mask = np.flipud(pole_mask)

        return ice,pole_mask

if __name__ == '__main__':
    '''Download data from FTP.'''

    from ftplib import FTP

    ftp_site = 'sidads.colorado.edu'
    ftp_dir = '/pub/DATASETS/nsidc0051_gsfc_nasateam_seaice/final-gsfc/{pole}/daily'

    diro = diri
    if not os.path.exists(diro):
        check_call(['mkdir','-p',diro])
    os.chdir(diro)

    pole = ['south','north']

    ftp = FTP(ftp_site)
    ftp.login()

    for p in pole:
        if not os.path.exists(p):
            check_call(['mkdir','-p',p])
        os.chdir(p)

        pwd1 = ftp.pwd()
        ftp.cwd(ftp_dir.format(pole=p))
        dirs = ftp.nlst()
        for d in dirs:
            if d == '.' or d == '..': continue
            ftp.cwd(d)
            files = ftp.nlst('*.bin')
            for filename in files:
                if os.path.exists(filename): continue
                with open(filename, 'wb') as fid:
                    print(f'transfering {filename}...',end='')
                    ftp.retrbinary(f'RETR {filename}', fid.write)
                    print('done.')
            ftp.cwd('..')
        ftp.cwd(pwd1)
        os.chdir(diro)

    ftp.cwd('/pub/DATASETS/seaice/polar-stereo/tools/')
    if not os.path.exists('polar-stereo-tools'):
        check_call(['mkdir','-p','polar-stereo-tools'])
    os.chdir('polar-stereo-tools')
    files = ftp.nlst('*.*')
    for filename in files:
        with open(filename, 'wb') as fid:
            print(f'transfering {filename}...',end='')
            ftp.retrbinary(f'RETR {filename}', fid.write)
            print('done.')

    ftp.quit()
