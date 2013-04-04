gpsd Navigator
==============

There are three scripts in this package.

navigator.py is for viewing current location, heading, and speed on the world
vector shoreline map.

navigation2img.py is similar. The difference is it exports several images in
different zoom levels and resolutions. Ideal for display on webpage.

readvct00.py is for converting the downloaded world vector shoreline in binary
format to a sqlite database which use by navigator.py and navigation2img.py.


License
=======

This package is released under the terms and conditions of the BSD License, a
copy of which is include in the file COPYRIGHT.


Requirement:
============

gpsd: The gps data is read from gpsd. Tested on gpsd 3.6, but other version
should also work.

Python: tested on python 2.7

Pygtk: it draws the GUI

Numpy: a coordinate is represented as a complex number which real portion is
the longitude, image portion is the latitude. Numpy makes transforming large
number of coordinates much easier. It is normally not installed by default,
search and install it from the repository of your distribution, or download and
install it from www.scipy.org

python-sqlite: used for store shoreline


How to run
==========

1. Download the world vector shoreline from NOAA.
   http://www.ngdc.noaa.gov/mgg/dat/geodas/coastlines/LittleEndian/wvs1mres.b00

2. Run readvct00.py to generate the shoreline sqlite database.

3. After gpsd started, run navigator.py or navigation2img.py.




Chen Wei <weichen302@gmx.com>
