# spec_utils
Some utility functions for infrared spectroscopy

# Requirements

## Functions
vgeo(obstime, obscoord, vhel=0, epoch=2000)

## Usage
```python
from astropy.time import Time
from astropy.coordinates import SkyCoord
from spec_utils import vgeo

mydate=Time('2014-08-16T00:00:00.0', format='isot', scale='utc')
mycoord=SkyCoord('16h31m33.46s', '-24d27m37.3s', frame='icrs')

#Calculate the heliocentric velocity (Earth-induced + intrinsic)
myv=vgeo(mydate, mycoord, vhel=-7.93)

print(myv, ' km/s')
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

