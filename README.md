# gpsd Navigator

There are three scripts in this package.

`navigator.py` is a gpsd client that plot current gps data (location, heading,
speed, etc.) on a world vector shoreline map. It can pan, zoom, rotate, measure
distance and heading like Goolge Earth does.  Pre-defined shapes such as lines
and placemarks can also be drawed on the map.

`navigation2img.py` is similar, but it exports several images in different zoom
levels and resolutions instead of draw on screen interactively. Ideal for
publish on web.

`readvct00.py` is for converting the binary NOAA world vector shoreline data to
a sqlite database which use by navigator.py and navigation2img.py.

## Screenshot

![Screenshot 1][]

![Screenshot 2][]


# License

This package is released under the terms and conditions of the BSD License, a
copy of which is include in the file COPYRIGHT.


# Requirement:

 * gpsd: The gps data is read from gpsd. Tested on gpsd 3.6, but other version
should also work.

 * Python: tested on python 2.7

 * Pygtk: it draws the GUI

 * [Numpy][]: a coordinate is represented as a complex number which real portion
  is the longitude, image portion is the latitude. Numpy makes transforming
large number of coordinates much easier. It is normally not installed by
default, search and install it from the repository of your distribution, or
download and install it from [scipy][].

 * python-sqlite: used for store shoreline


# How to run

1. Download the [world vector shoreline][] from NOAA.

2. To generate the shoreline sqlite database, run:

    $ readvct00.py -i inputfile -o output-sqlite

3. After gpsd started, run `navigator.py` or `navigation2img.py`.



[Contact me](mailto: weichen302@gmx.com)

[world vector shoreline]: http://www.ngdc.noaa.gov/mgg/dat/geodas/coastlines/LittleEndian/wvs1mres.b00
[Screenshot 1]: https://raw.github.com/infinet/gpsd-navigator/gh-pages/images/screenshot_3.png
[Screenshot 2]: https://raw.github.com/infinet/gpsd-navigator/gh-pages/images/screenshot_4.png
[gpsd]: http://www.catb.org/gpsd/
[scipy]: http://www.scipy.org


