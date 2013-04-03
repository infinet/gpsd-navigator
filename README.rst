gpsd Navigator
==============

There are three scripts in this package.

navigator.py is for viewing current location, heading, and speed on the world
vector shoreline map.

navigation2img.py is similar. The difference is it export several images in
different zoom levels and resolutions. Ideal for display on webpage.

readvct00.py is for converting the downloaded world vector shoreline in binary
format to a sqlite database file. Navigator and navigation2img.py query this
database file to retrieve the shoreline data.

License
=======

This package is released under the terms and conditions of the BSD License, a
copy of which is include in the file COPYRIGHT.


Requirement:
------------

It gets GPS information from gpsd, a daemon connect to a GPS device, by USB,
Bluetooth, or RS232. Any linux distribution comes with a recent version of
python and gpsd should be able to run it. There are gpsd for MacOS X and Windows
too, try it youself.

python: tested on python 2.7

pygtk: it draws the GUI

gpsd: download and install it from http://www.catb.org/gpsd/ if not found in
your system.

numpy: a coordinate is processed as a complex number which real portion is the
longitude, image portion is the latitude. Numpy makes transform large number of
coordinates much easier. It is normally not installed by default, search and
install it from the repository of your distribution, or download and install it
from www.scipy.org

python-sqlite: used for store shoreline

How to run
----------

1) download the world vector shoreline
http://www.ngdc.noaa.gov/mgg/dat/geodas/coastlines/LittleEndian/wvs1mres.b00

2) run readvct00.py to generate the shoreline sqlite database.

3) after gpsd is started, run navigator.py or navigation2img.py.




Chen Wei <weichen302@gmx.com>
