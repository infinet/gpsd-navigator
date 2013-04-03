#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
gpsd navigator image exporter
similar to gpsd navigator, but export current location and the world vector map
as images in several zoom levle and resolution. It is intended to generate the
images for webserver.
'''

__license__ = 'BSD'
__copyright__ = '2013, Chen Wei <weichen302@gmx.com>'
__version__ = '0.0.2'

from math import pi, sin, cos, asin, sqrt, radians
from socket import error as SocketError
import cairo
import cmath
import gps
import math
import numpy as np
import os
import sqlite3
import sys
import time

WIN_Y = 800
WIN_X = 1300
MOBILE_X = 960  # for cellphone screen, iphone 4 uses 960x640
MOBILE_Y = 640

TRACKBUFMAX = 6000
CMPS_SIZE = 40
CMPS_N_SIZE = 5
EARTH_R = 6371009   # in meters
ROTATE = 0
SCALE = 50
DELAY_UPDATE = 0.3  # time waiting for update after mouse zoom
MAX_ZOOMLEVEL = 0.06  # max zoom out level, based on test and try
scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
UNIT = {'Mph': gps.MPS_TO_MPH,
        'Kmh': gps.MPS_TO_KPH,
        'Knots': gps.MPS_TO_KNOTS,
        'Meters': 1.0, 'Kilometers': 0.001,
        'Nautical Miles': 1 / 1855.325, 'Miles': 1 / 1609.344}

navconf = os.path.join(scriptpath, 'navigation.conf')
DB_SHORELINE = os.path.join(scriptpath, 'shoreline.sqlite')

WWW_ROOT = '/tmp'
TIMEOUT_ZOOM_1 = 313
TIMEOUT_ZOOM_2 = 297
TIMEOUT_ZOOM_3 = 37
TIMEOUT_ZOOM_4 = 13
TIMEOUT_ZOOM_5 = 5

# [zoomlevel, self.ref]
ZOOM_PARM = {'zoom1': [0.078, complex(50.0, 0)],
             'zoom2': [0.143, complex(20.0, 12.0)],
             'zoom3': [0.428, complex(-17.0, 21.0)],
             'zoom4': [10.0, None],
             'zoom5': [17.0, None]}

ZOOMS = ('zoom1', 'zoom2', 'zoom3', 'zoom4', 'zoom5')


def get_config():
    '''parse the configure file, return a dictionary of preconfigured
    shapes and locations'''
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.readfp(open(navconf))

    res = {'qagl': {'color': [], 'cord': []},
           'line': {'color': [], 'cord': []},
           'center': complex(0, 0),
           'pts': []}
    for sec in config.sections():
        if 'point' in sec:
            pt = {'name': config.get(sec, 'name'),
                  'cord': complex(float(config.get(sec, 'lon')),
                                  float(config.get(sec, 'lat')))}
            res['pts'].append(pt)

        elif 'center' in sec:
            res['center'] = complex(float(config.get(sec, 'lon')),
                                    float(config.get(sec, 'lat')))

        elif 'quadrangle' in sec:
            qagl = [get_config_proc(config.get(sec, 'SW')),
                    get_config_proc(config.get(sec, 'SE')),
                    get_config_proc(config.get(sec, 'NE')),
                    get_config_proc(config.get(sec, 'NW'))]
            res['qagl']['cord'].append(qagl)
            res['qagl']['color'].append(
                [float(x) for x in config.get(sec, 'color').split(',')])

        elif 'line' in sec:
            res['line']['color'].append(
                [float(x) for x in config.get(sec, 'color').split(',')])
            res['line']['cord'].append(
                                [get_config_proc(config.get(sec, 'start')),
                                 get_config_proc(config.get(sec, 'end'))])
    if res['line']['cord']:
        res['line']['cord'] = np.array(res['line']['cord'])
    if res['qagl']['cord']:
        res['qagl']['cord'] = np.array(res['qagl']['cord'])

    return res


def get_config_proc(line):
    '''parse config line such as -17.751794, 21.471444 into complex number'''
    line = [float(x) for x in line.split(',')]
    return complex(line[0], line[1])


def get_depth(fname):
    '''get water depth, return a 3 x 196 x Y list'''
    fp = open(fname)
    res = []
    pencode = []
    for line in fp:
        line = line.split(',')
        res.append(complex(float(line[0]), float(line[1])))
        pencode.append(int(line[2]))
    return (np.array(res), pencode)


def earthdistance(c1, c2):
    '''given two WGS84 coordinates in complex number, calculate distance,
    use haversine formula for small distance
    http://en.wikipedia.org/wiki/Great-circle_distance
    '''
    delta_lon = radians(c1.real - c2.real)
    delta_lat = radians(c1.imag - c2.imag)
    theta = 2 * asin(sqrt(sin(delta_lat / 2) ** 2 +
     cos(radians(c1.imag)) * cos(radians(c2.imag)) * sin(delta_lon / 2) ** 2))
    return theta * EARTH_R


def degree2dms(degree, category='longitude'):
    """convert a degree to degree minutes' seconds'' """
    if category == 'longitude':
        postfix = 'E' if degree >= 0 else 'W'
    elif category == 'latitude':
        postfix = 'N' if degree >= 0 else 'S'
    degree = math.fabs(degree)
    tmp, deg = math.modf(degree)
    minutes = tmp * 60
    secs = math.modf(minutes)[0] * 60
    res = '''%d%s%d'%.2f"%s''' % (int(deg), u'\N{DEGREE SIGN}',
                                  math.floor(minutes), secs, postfix)
    return res


class Navigation():
    '''the main part'''

    def __init__(self):
        self.rotate = ROTATE
        self.timer = {'zoom1': [0, TIMEOUT_ZOOM_1],
                      'zoom2': [0, TIMEOUT_ZOOM_2],
                      'zoom3': [0, TIMEOUT_ZOOM_3],
                      'zoom4': [0, TIMEOUT_ZOOM_4],
                      'zoom5': [0, TIMEOUT_ZOOM_5]}

        self.zoomlevel = 10
        self.position = 0
        self.track = np.zeros(TRACKBUFMAX, dtype='complex')
        self.track_indx = 0
        self.track_rewind = False
        self.track_refresh_cnt = 0
        self.heading = 0  # in radian
        self.dialog = False  # status of popup dialog
        self.flag_ruler_start = self.flag_ruler_end = False
        self.unitfactor = 1  # convert between meter/km/nm/mile
        self.speed_unit = 'Knots'
        self.last_speed = 0
        self.utc_time = ''
        self.heading_degree = 0
        self.ruler_distance = 0
        self.size_x = WIN_X
        self.size_y = WIN_Y
        self.show_placemarks = True
        self.placemarks = get_config()
        # lon and latitude of center, in numpy complex number array
        self.ref = np.array([self.placemarks['center']])[0]
        self.db = sqlite3.connect(DB_SHORELINE).cursor()
        self.pencode = {}
        self.shoreline = {}
        for zoom in ZOOMS:
            self.zoomlevel = ZOOM_PARM[zoom][0]
            if ZOOM_PARM[zoom][1]:
                self.ref = self.center_location(self.zoomlevel,
                                                ZOOM_PARM[zoom][1])
                self.refresh_shoreline(zoom)

    def expose_mobile(self, zoom):
        self.size_x, self.size_y = MOBILE_X, MOBILE_Y
        # Create the cairo context
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, MOBILE_X, MOBILE_Y)
        self.cr = cairo.Context(surface)
        #self.cr = self.window.cairo_create()
        self.cr.set_source_rgb(0, 0, 0)
        self.cr.rectangle(0, 0, MOBILE_X, MOBILE_Y)
        self.cr.fill()
        # Restrict Cairo to the exposed area; avoid extra work
        self.cr.rectangle(0, 0, MOBILE_X, MOBILE_Y)
        self.cr.clip()

        if self.show_placemarks:
            self.draw_placemarks()

        self.draw_lines()
        self.draw_quadrangle()
        self.draw_shoreline(zoom)
        if self.position:
            self.draw_position()
            self.draw_track()
        #self.draw_compass()
        self.draw_stdruler()
        self.draw_status()
        fname = os.path.join(WWW_ROOT, 'mobile', 'image', zoom + '.png')
        surface.write_to_png(fname)
        print 'Update mobile map for zoom level %s' % zoom[-1]

    def expose_event(self, zoom):
        self.size_x, self.size_y = WIN_X, WIN_Y
        # Create the cairo context
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.size_x, self.size_y)
        self.cr = cairo.Context(surface)
        #self.cr = self.window.cairo_create()
        self.cr.set_source_rgb(0, 0, 0)
        self.cr.rectangle(0, 0, self.size_x, self.size_y)
        self.cr.fill()
        # Restrict Cairo to the exposed area; avoid extra work
        self.cr.rectangle(0, 0, self.size_x, self.size_y)
        self.cr.clip()

        if self.show_placemarks:
            self.draw_placemarks()

        self.draw_lines()
        self.draw_quadrangle()
        self.draw_shoreline(zoom)
        if self.position:
            self.draw_position()
            self.draw_track()
        self.draw_compass()
        self.draw_stdruler()
        self.draw_status()
        fname = os.path.join(WWW_ROOT, 'image', zoom + '.png')
        surface.write_to_png(fname)
        print 'Update map for zoom level %s' % zoom[-1]
#         print 'current zoomlevel is %f' % self.zoomlevel
        # print self.ref

    def queue_draw(self):
        '''burrow the name from gtk so can use same piece of code in Main'''
        for zoom in ZOOMS:
            now = time.time()
            if now - self.timer[zoom][0] > self.timer[zoom][1]:
                self.zoomlevel = ZOOM_PARM[zoom][0]
                if ZOOM_PARM[zoom][1] != None:
                    self.ref = self.center_location(self.zoomlevel,
                                                    ZOOM_PARM[zoom][1])
                    self.ref_mobile = self.center_location(self.zoomlevel,
                                                    self.position,
                                                    size_x=MOBILE_X,
                                                    size_y=MOBILE_Y)

                else:
                    # refresh the current location to the center of image
                    print '\nCalc map center for zoom level %s' % zoom[-1]
                    self.ref = self.center_gps(ZOOM_PARM[zoom][0])
                    self.ref_mobile = self.center_location(self.zoomlevel,
                                                    self.position,
                                                    size_x=MOBILE_X,
                                                    size_y=MOBILE_Y)
                    self.refresh_shoreline(zoom)
                self.timer[zoom][0] = time.time()
                self.expose_event(zoom)

                # change self.ref for mobile
                self.ref = self.ref_mobile
                self.expose_mobile(zoom)

    def center_gps(self, zl):
        '''calculate the self.ref for a given zoomlevel so the current
        location will show in the center of screen'''

        k = cmath.rect(zl * SCALE, self.rotate)
        return self.position - complex(self.size_x, self.size_y) / (2 * k)

    def center_location(self, zl, loc, size_x=WIN_X, size_y=WIN_Y):
        '''calculate the self.ref for a given zoomlevel and location, so that
        location will show in the center of screen'''

        k = cmath.rect(zl * SCALE, self.rotate)
        return loc - complex(size_x, size_y) / (2 * k)

    def draw_placemarks(self):
        '''draw placemarks'''
        self.cr.set_line_width(1)

        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)

        # draw locations (points)
        for pt in self.placemarks['pts']:
            self.cr.set_source_rgb(1, 0.7, 0.3)
            arc_r = 5
            loc = (pt['cord'] - self.ref) * k
            ref_x, ref_y = loc.real, self.size_y - loc.imag
            self.cr.arc(ref_x, ref_y, arc_r, 0, 2 * pi)
            self.cr.close_path()
            self.cr.fill()
            self.draw_text(ref_x + 10, ref_y, pt['name'], fontsize=12,
                          align='left')

        self.cr.stroke()

    def xy2wgs84(self, x, y):
        '''convert a point(x, y) of gtk screen to WGS84 coordinate
        Return:
            a coordinate in complex number'''
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        return complex(x, self.size_y - y) / k + self.ref

    def draw_ruler(self):
        p_start = self.ruler_start - self.ref
            #self.ruler_start = False
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        p_start *= k
        self.cr.set_source_rgba(1, 1, 0, 0.8)
        self.cr.set_line_width(1)
        self.cr.move_to(p_start.real, self.size_y - p_start.imag)
        if self.flag_ruler_end:
            x, y = self._mouseX, self._mouseY
        else:
            x, y = self.pointer_x, self.pointer_y
        self.cr.line_to(x, y)
        self.cr.stroke()

    def draw_stdruler(self):
        start_x, start_y = self.size_x - 240, self.size_y - 60
        m = 50
        self.cr.set_line_width(0.5)
        self.cr.set_source_rgba(1, 1, 1, 0.8)
        tickunit = 4
        points = []
        ticks = []
        for i in xrange(5):
            if i % 4 == 0:
                ticklen = 3
            elif i == 2:
                ticklen = 2
            else:
                ticklen = 1
            points.append([start_x + i * m, start_y])
            ticks.append([start_x + i * m, start_y - ticklen * tickunit])
        for i in xrange(5):
            self.cr.move_to(*points[i])
            self.cr.line_to(*ticks[i])

        self.cr.move_to(*points[0])
        self.cr.line_to(*points[-1])
        self.cr.stroke()

        c1 = self.xy2wgs84(*points[0])
        c2 = self.xy2wgs84(*points[-1])
        distance = earthdistance(c1, c2)
        if distance > 5000:
            txt = '%.2f Km' % (distance / 1000)
        else:
            txt = '%d m' % int(distance)
        self.draw_text(ticks[-1][0], ticks[-1][1] - 5, txt, fontsize=12)

    def draw_status(self):
        '''draw current lon/lat under mouse pointer on the bottom statusline'''
        cord = self.position
        speed = self.last_speed * UNIT[self.speed_unit]
        hstxt = '%3d    %.2f %s    %s' % (self.heading_degree, speed,
                                        self.speed_unit, self.utc_time)

        cordtxt = '%s  %s' % (degree2dms(cord.real, category='longitude'),
                              degree2dms(cord.imag, category='latitude'))
        self.cr.set_line_width(0.5)
        self.cr.set_source_rgba(1, 1, 1, 0.8)
        self.draw_text(self.size_x - 130, self.size_y - 40,
                                                   cordtxt, fontsize=12)
        self.draw_text(10, 10, hstxt, fontsize=12, align='left',
                      color=((0.745, 0.812, 0.192, 1)))

    def draw_text(self, x, y, text, fontsize=10, align='center',
                 color=(1, 1, 1, 0.8)):
        '''draw text at given location
        Args:
            x, y is the center of textbox'''
        #txt = str(text)
        txt = text
        self.cr.new_sub_path()
        self.cr.set_source_rgba(*color)
        self.cr.select_font_face('Sans')
        self.cr.set_font_size(fontsize)
        (x_bearing, y_bearing,
         t_width, t_height) = self.cr.text_extents(txt)[:4]
        # set the center of textbox
        if align == 'center':
            self.cr.move_to(x - t_width / 2, y + t_height / 2)
        elif align == 'left':
            self.cr.move_to(x, y + t_height / 2)
        else:
            self.cr.move_to(x + t_width / 2, y + t_height / 2)

        self.cr.show_text(txt)

    def refresh_shoreline(self, zoom):
        '''refresh shoreline data from sqlite or local file'''
        print 'refresh shoreline for zoom level %s' % zoom[-1]
        p1 = self.xy2wgs84(0, self.size_y)
        p2 = self.xy2wgs84(self.size_x, 0)
        p3 = self.xy2wgs84(0, 0)
        p4 = self.xy2wgs84(self.size_x, self.size_y)
        lon_min = min(p1.real, p2.real, p3.real, p4.real) - 2
        lon_max = max(p1.real, p2.real, p3.real, p4.real) + 2
        lat_min = min(p1.imag, p2.imag, p3.imag, p4.imag) - 2
        lat_max = max(p1.imag, p2.imag, p3.imag, p4.imag) + 2

        if self.zoomlevel < 0.1:
            resolution_level = 7
        elif self.zoomlevel < 0.2:
            resolution_level = 6
        elif self.zoomlevel < 0.5:
            resolution_level = 5
        elif self.zoomlevel < 1:
            resolution_level = 4
        elif self.zoomlevel < 2:
            resolution_level = 3
        elif self.zoomlevel < 5:
            resolution_level = 2
        else:
            resolution_level = 1

        #resolution_level = 7   # debug
        #self.timing['db_startquery'] = time.time()
        sqlshore = ('SELECT lon, '
                            'lat, '
                            'penstart, '
                            'segment '
                    'FROM shore '
                    'WHERE lon > ? '
                            'AND lat > ? '
                            'AND lon < ? '
                            'AND lat < ? '
                            'AND res%d = ?') % resolution_level
        self.db.execute(sqlshore,
                        (lon_min, lat_min, lon_max, lat_max, 1))
        rows = self.db.fetchall()
        #print "total %d points" % len(rows)
        self.pencode[zoom] = [x[2:] for x in rows]
        self.shoreline[zoom] = np.array([complex(x[0], x[1]) for x in rows])

        # debug
        #t = time.time() - self.timing['db_startquery']
        #print 'sqlite query used %f' % t
        #print 'Zoom Level = %f, %d points' % (self.zoomlevel, len(rows))

    def draw_shoreline(self, zoom):
        '''read shoreline coordinates generated from sql query and draw it'''

        self.cr.set_line_width(1)
        self.cr.set_source_rgb(0, 0.7, 0.3)
        try:
            shoreline = self.shoreline[zoom] - self.ref
            pencode = self.pencode[zoom]
        except KeyError:
            return
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        shoreline *= k

        self.cr.move_to(shoreline[0].real, self.size_y - shoreline[0].imag)
        # self.timing['shore_startdraw'] = time.time()
        segid = 0
        for i in xrange(1, len(pencode)):
            if pencode[i][0] == 1:
                segid = pencode[i][1]
                self.cr.move_to(shoreline[i].real,
                                self.size_y - shoreline[i].imag)
            elif pencode[i][1] == segid:
                self.cr.line_to(shoreline[i].real,
                                self.size_y - shoreline[i].imag)
        self.cr.stroke()

    def draw_lines(self):
        '''draw lines'''
        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        cords = (self.placemarks['line']['cord'] - self.ref) * k
        # draw locations (points)
        self.cr.set_line_width(1)
        #self.cr.set_source_rgb(0, 1, 1)
        for i in xrange(len(self.placemarks['line']['cord'])):
            lcolor = self.placemarks['line']['color'][i]
            cord = cords[i]
            self.cr.set_source_rgba(*lcolor)
            self.cr.move_to(cord[0].real, self.size_y - cord[0].imag)
            self.cr.line_to(cord[1].real, self.size_y - cord[1].imag)

        self.cr.stroke()

    def draw_quadrangle(self):
        '''draw lines'''
        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        cords = (self.placemarks['qagl']['cord'] - self.ref) * k
        self.cr.set_line_width(1)
        for i in xrange(len(self.placemarks['qagl']['cord'])):
            lcolor = self.placemarks['qagl']['color'][i]
            cord = cords[i]
            self.cr.set_source_rgba(*lcolor)
            self.cr.move_to(cord[0].real, self.size_y - cord[0].imag)
            self.cr.line_to(cord[1].real, self.size_y - cord[1].imag)
            self.cr.line_to(cord[2].real, self.size_y - cord[2].imag)
            self.cr.line_to(cord[3].real, self.size_y - cord[3].imag)
            self.cr.line_to(cord[0].real, self.size_y - cord[0].imag)

        self.cr.stroke()

    def draw_track(self):
        self.cr.set_line_width(3)
        self.cr.set_source_rgb(0.882, 0.145, 0.647)
        track = self.track - self.ref
        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        track *= k
        # project to cairio x, y coordinates
        self.cr.move_to(track[0].real, self.size_y - track[0].imag)
        i = 1
        while i < self.track_indx:
            self.cr.line_to(track[i].real, self.size_y - track[i].imag)
            i += 1
        if self.track_rewind:
            print 'rewinded, track index= %d' % self.track_indx
            rwindx = self.track_indx + 1
            if rwindx < TRACKBUFMAX:
                self.cr.move_to(track[rwindx].real,
                                self.size_y - track[rwindx].imag)
            rwindx += 1
            while rwindx < TRACKBUFMAX:
                self.cr.line_to(track[rwindx].real,
                                self.size_y - track[rwindx].imag)
                rwindx += 1
        self.cr.stroke()

    def draw_position(self):
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        arc_r = 50
        loc = (self.position - self.ref) * k
        ref_x, ref_y = loc.real, self.size_y - loc.imag
        self.cr.arc(ref_x, ref_y, 2, 0, 2 * pi)
        self.cr.fill()
        self.cr.set_source_rgb(1.0, 0, 0)
        self.cr.arc(ref_x, ref_y, arc_r, 0, 2 * pi)
        self.cr.stroke()

        # draw the trig
        # vector of heading
        vh = cmath.rect(arc_r, self.heading) * cmath.rect(1, self.rotate)
        # Top point of trig
        vt = loc + vh
        # Bottom point
        trig_size = 10
        delta = cmath.rect(trig_size, cmath.phase(vh) + pi / 6)
        vb = delta * cmath.rect(1, pi)  # rotate 180
        va = delta * cmath.rect(1, pi * 2 / 3)
        vb = vt + vb
        va = vt + va
        self.cr.move_to(ref_x, ref_y)
        self.cr.line_to(vt.real, self.size_y - vt.imag)
        self.cr.stroke()
        self.cr.line_to(vb.real, self.size_y - vb.imag)
        self.cr.line_to(va.real, self.size_y - va.imag)
        self.cr.line_to(vt.real, self.size_y - vt.imag)
        self.cr.close_path()
        self.cr.fill()

    def draw_compass(self):
        '''draw a compass'''
        # out circle
        self.cr.set_source_rgba(1, 1, 1, 0.7)
        self.cr.set_line_width(3)
        self.cr.arc(self.size_x - 60, 60, CMPS_SIZE, 0, 2 * pi)
        self.cr.stroke()

        # position of compass pointer
        nloc = cmath.rect(CMPS_SIZE, self.rotate + pi / 2)
        x, y = nloc.real + self.size_x - 60, nloc.imag + self.size_y - 60
        self.cr.arc(x, self.size_y - y, CMPS_N_SIZE + 3, 0, 2 * pi)
        self.cr.close_path()
        self.cr.fill()

        # draw N
        self.cr.set_source_rgb(0, 0, 0)
        shape_n = np.array([cmath.rect(CMPS_N_SIZE, self.rotate + pi * 5 / 4),
                            cmath.rect(CMPS_N_SIZE, self.rotate + pi * 3 / 4),
                            cmath.rect(CMPS_N_SIZE, self.rotate + pi * 7 / 4),
                            cmath.rect(CMPS_N_SIZE, self.rotate + pi * 1 / 4)])

        # move to the location of compass pointer
        shape_n += complex(x, y)
        self.cr.move_to(shape_n[0].real, self.size_y - shape_n[0].imag)
        for point in shape_n[1:]:
            self.cr.line_to(point.real, self.size_y - point.imag)
        self.cr.stroke()

    def get_x_y(self):
        rect = self.get_allocation()
        x = (rect.x + rect.width / 2.0)
        y = (rect.y + rect.height / 2.0) - 20
        return x, y

    def run(self):
        while True:
            print self.position
            self.queue_draw()
            time.sleep(1)


class Main(object):
    def __init__(self, host='localhost', port='2947', device=None, debug=0):
        self.host = host
        self.port = port
        self.device = device
        self.debug = debug
        self.widget = Navigation()
        self.newpt_count = 0

    def handle_response(self, source):
        print 'in handle repsonse'
        if self.daemon.read() == -1:
            self.handle_hangup(source)
        if self.daemon.data['class'] == 'TPV':
            self.update_speed(self.daemon.data)
        if self.daemon.data['class'] == 'SKY':
            self.update_skyview(self.daemon.data)

        return True

    def update_speed(self, data):
        '''put image exporting control here, use a timer'''
        if hasattr(data, 'time'):
            tstr = [c for c in data.time[:-5]]
            tstr[10] = ' '
            tstr.append(' UTC')
            self.widget.utc_time = ''.join(tstr)

        if hasattr(data, 'speed'):
            self.widget.last_speed = data.speed
        if hasattr(data, 'track'):
            self.widget.heading = (90 - int(data.track)) * pi / 180
            self.widget.heading_degree = int(data.track)
        if hasattr(data, 'lon') and hasattr(data, 'lat'):
            pos = complex(float(data.lon), float(data.lat))
            self.newpt_count += 1
            distance = earthdistance(pos,
                             self.widget.track[self.widget.track_indx - 1])
            #print '\ndistance between points %f m' % distance
            # update position if distance greater than 10m, or every
            # 20 gps reading received
            if (self.newpt_count > 20) or (distance > 20):
                self.newpt_count = 0
                self.widget.position = pos
                if self.widget.track_indx < TRACKBUFMAX:
                    track_indx = self.widget.track_indx
                    self.widget.track_indx += 1
                    print 'track index is %d' % self.widget.track_indx
                else:
                    # reache the end of track numpy array, rewind
                    print 'i am here, rewinding'
                    track_indx = self.widget.track_indx = 0
                    self.widget.track_rewind = True
                self.widget.track[track_indx] = pos

    def run(self):
        try:
            session = gps.gps(host=self.host, port=self.port)
            session.stream(gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)

            while True:
                rpt = session.next()
                if rpt['class'] == 'TPV':
                    self.update_speed(rpt)
                    self.widget.queue_draw()
        except StopIteration:
            print 'stop iteration'
        except SocketError:
            print 'could not connect to gpsd socket. Is gpsd running?'
        except KeyboardInterrupt:
            print 'bye'

if __name__ == "__main__":
    import sys
    from os.path import basename
    from optparse import OptionParser
    prog = basename(sys.argv[0])
    usage = ('%s [--host] ' +
            '[--port] [--device] ' +
            '[host [:port [:device]]]') % (prog)

    parser = OptionParser(usage=usage)
    parser.add_option(
            '--host',
            dest='host',
            default='localhost',
            help='The host to connect. [Default localhost]'
    )
    parser.add_option(
            '--port',
            dest='port',
            default='2947',
            help='The port to connect. [Default 2947]'
    )
    parser.add_option(
            '--device',
            dest='device',
            default=None,
            help='The device to connet. [Default None]'
    )
    (options, args) = parser.parse_args()
    if args:
        arg = args[0].split(':')
        len_arg = len(arg)
        if len_arg == 1:
            (options.host,) = arg
        elif len_arg == 2:
            (options.host, options.port) = arg
        elif len_arg == 3:
            (options.host, options.port, options.device) = arg
        else:
            parser.print_help()
            sys.exit(0)
    Main(host=options.host,
            port=options.port,
            device=options.device
            ).run()
