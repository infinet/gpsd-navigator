#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
gpsd navigator
show current location on the world vector shoreline map
'''

__license__ = 'BSD'
__copyright__ = '2013, Chen Wei <weichen302@gmx.com>'
__version__ = '0.0.2'

from math import pi, sin, cos, asin, sqrt, radians
from socket import error as SocketError
import cmath
import gobject
import gps
import gtk
import gtk.gdk as gdk
import math
import numpy as np
import os
import pygtk
import sqlite3
import sys
import time
pygtk.require('2.0')

WIN_Y = 800
WIN_X = 1300
CMPS_SIZE = 40
CMPS_N_SIZE = 5
EARTH_R = 6371009   # in meters
ROTATE = 0
SCALE = 50
DELAY_UPDATE = 0.3  # time waiting for update after mouse zoom
MAX_ZOOMLEVEL = 0.06  # max zoom out level, based on test and try
UNIT = {'Mph': gps.MPS_TO_MPH,
        'Kmh': gps.MPS_TO_KPH,
        'Knots': gps.MPS_TO_KNOTS,
        'Meters': 1.0, 'Kilometers': 0.001,
        'Nautical Miles': 1 / 1855.325, 'Miles': 1 / 1609.344}

scriptpath = os.path.abspath(os.path.dirname(sys.argv[0]))
navconf = os.path.join(scriptpath, 'navigation.conf')
DB_SHORELINE = os.path.join(scriptpath, 'shoreline.sqlite')


def get_config():
    '''parse the configure file, return a dictionary of preconfigured
    shapes and locations'''
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.readfp(open(navconf))

    res = {'qagl': {'color': [], 'cord': []},
           'line': {'color': [], 'cord': []},
           'center': complex(0, 0),
           'pts': [],
           'gpsd-server': {}}
    for sec in config.sections():
        if 'gpsd-server' in sec:
            res['gpsd-server']['host'] = config.get(sec, 'host')
            res['gpsd-server']['port'] = config.get(sec, 'port')

        elif 'point' in sec:
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


class Navigation(gtk.DrawingArea):
    '''the main part'''

    def __init__(self):
        gtk.DrawingArea.__init__(self)
        self.connect('expose_event', self.expose_event)
        self.connect("button_press_event", self._mouse_press)
        self.connect("button_release_event", self._mouse_release)
        self.connect("scroll_event", self._mouseScroll)
        self.connect("motion_notify_event", self._mouse_move)
        self.set_events(self.get_events()
                                        | gdk.BUTTON_PRESS_MASK
                                        | gdk.BUTTON_RELEASE_MASK
                                        | gdk.POINTER_MOTION_MASK
                                        | gdk.POINTER_MOTION_HINT_MASK)
        self._mouseX = self._mouseY = 0
        self.pointer_x = self.pointer_y = 0
        self._dragPosX = self._dragPosY = self._dragPosZ = 0.
        self._dragDeltaX = self._dragDeltaY = self._dragDeltaZ = 0.
        self.rotate = ROTATE
        self.zoomlevel = 10
        self.position = 0
        self.track = np.zeros(8640, dtype='complex')
        self.track_indx = 0
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
        self.timing = {'mouse_zoom': 0, 'update_waiting': False,
                       'last_zoom': 0}
        self.show_placemarks = True
        self.placemarks = get_config()
        # lon and latitude of center, in numpy complex number array
        self.ref = np.array([self.placemarks['center']])[0]
        self.db = sqlite3.connect(DB_SHORELINE).cursor()
        self.refresh_shoreline()

    def expose_event(self, widget, event, data=None):
        # Create the cairo context
        self.cr = self.window.cairo_create()
        rectx, recty = self.get_x_y()
        self.cr.set_source_rgb(0, 0, 0)
        self.cr.rectangle(0, 0, rectx * 2, recty * 2)
        self.cr.fill()
        # Restrict Cairo to the exposed area; avoid extra work
        self.cr.rectangle(event.area.x, event.area.y,
                event.area.width, event.area.height)
        self.cr.clip()

        if self.show_placemarks:
            self.draw_placemarks()

        self.draw_lines()
        self.draw_quadrangle()
        self.draw_shoreline()
        if self.position:
            self.draw_position()
            self.draw_track()
        self.draw_compass()
        self.draw_stdruler()
        self.draw_status(self.pointer_x, self.pointer_y)
        if self.dialog and (self.flag_ruler_start or self.flag_ruler_end):
            self.draw_ruler()
        # print 'current zoomlevel is %f' % self.zoomlevel

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
            ref_x, ref_y = loc.real, WIN_Y - loc.imag
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
        return complex(x, WIN_Y - y) / k + self.ref

    def draw_ruler(self):
        p_start = self.ruler_start - self.ref
            #self.ruler_start = False
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        p_start *= k
        self.cr.set_source_rgba(1, 1, 0, 0.8)
        self.cr.set_line_width(1)
        self.cr.move_to(p_start.real, WIN_Y - p_start.imag)
        if self.flag_ruler_end:
            x, y = self._mouseX, self._mouseY
        else:
            x, y = self.pointer_x, self.pointer_y
        self.cr.line_to(x, y)
        self.cr.stroke()

    def draw_stdruler(self):
        start_x, start_y = WIN_X - 240, WIN_Y - 60
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

    def draw_status(self, x, y):
        '''draw current lon/lat under mouse pointer on the bottom statusline'''
        cord = self.xy2wgs84(x, y)
        speed = self.last_speed * UNIT[self.speed_unit]
        hstxt = '%3d    %.2f %s    %s' % (self.heading_degree, speed,
                                        self.speed_unit, self.utc_time)

        cordtxt = '%s  %s' % (degree2dms(cord.real, category='longitude'),
                              degree2dms(cord.imag, category='latitude'))
        self.cr.set_line_width(0.5)
        self.cr.set_source_rgba(1, 1, 1, 0.8)
        self.draw_text(WIN_X - 130, WIN_Y - 40, cordtxt, fontsize=12)
        self.draw_text(10, WIN_Y - 40, hstxt, fontsize=12, align='left',
                      color=((0.745, 0.812, 0.192, 1)))

    def draw_text(self, x, y, text, fontsize=10, align='center',
                 color=(1, 1, 1, 0.8)):
        '''draw text at given location
        Args:
            x, y is the center of textbox'''
        txt = str(text)
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

    def refresh_shoreline(self):
        '''refresh shoreline data from sqlite or local file'''

        p1 = self.xy2wgs84(0, WIN_Y)
        p2 = self.xy2wgs84(WIN_X, 0)
        p3 = self.xy2wgs84(0, 0)
        p4 = self.xy2wgs84(WIN_X, WIN_Y)
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
        self.pencode = [x[2:] for x in rows]
        self.shoreline = np.array([complex(x[0], x[1]) for x in rows])

        # debug
        #t = time.time() - self.timing['db_startquery']
        #print 'sqlite query used %f' % t
        #print 'Zoom Level = %f, %d points' % (self.zoomlevel, len(rows))

    def draw_shoreline(self):
        '''read shoreline coordinates generated from sql query and draw it'''

        self.cr.set_line_width(1)
        self.cr.set_source_rgb(0, 0.7, 0.3)
        shoreline = self.shoreline - self.ref
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        shoreline *= k

        self.cr.move_to(shoreline[0].real, WIN_Y - shoreline[0].imag)
        # self.timing['shore_startdraw'] = time.time()
        segid = 0
        for i in xrange(1, len(self.pencode)):
            if self.pencode[i][0] == 1:
                segid = self.pencode[i][1]
                self.cr.move_to(shoreline[i].real, WIN_Y - shoreline[i].imag)
            elif self.pencode[i][1] == segid:
                self.cr.line_to(shoreline[i].real, WIN_Y - shoreline[i].imag)
        self.cr.stroke()

    def draw_lines(self):
        '''draw lines'''
        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        cords = (self.placemarks['line']['cord'] - self.ref) * k
        self.cr.set_line_width(1)
        for i in xrange(len(self.placemarks['line']['cord'])):
            lcolor = self.placemarks['line']['color'][i]
            cord = cords[i]
            self.cr.set_source_rgba(*lcolor)
            self.cr.move_to(cord[0].real, WIN_Y - cord[0].imag)
            self.cr.line_to(cord[1].real, WIN_Y - cord[1].imag)

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
            self.cr.move_to(cord[0].real, WIN_Y - cord[0].imag)
            self.cr.line_to(cord[1].real, WIN_Y - cord[1].imag)
            self.cr.line_to(cord[2].real, WIN_Y - cord[2].imag)
            self.cr.line_to(cord[3].real, WIN_Y - cord[3].imag)
            self.cr.line_to(cord[0].real, WIN_Y - cord[0].imag)

        self.cr.stroke()

    def draw_track(self):
        self.cr.set_line_width(3)
        self.cr.set_source_rgb(0.882, 0.145, 0.647)
        track = self.track - self.ref
        # zoom and scale
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        track *= k
        # project to cairio x, y coordinates
        self.cr.move_to(track[0].real, WIN_Y - track[0].imag)
        for i in xrange(self.track_indx):
            self.cr.line_to(track[i].real, WIN_Y - track[i].imag)
        self.cr.stroke()

    def draw_position(self):
        k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        arc_r = 50
        loc = (self.position - self.ref) * k
        ref_x, ref_y = loc.real, WIN_Y - loc.imag
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
        self.cr.line_to(vt.real, WIN_Y - vt.imag)
        self.cr.stroke()
        self.cr.line_to(vb.real, WIN_Y - vb.imag)
        self.cr.line_to(va.real, WIN_Y - va.imag)
        self.cr.line_to(vt.real, WIN_Y - vt.imag)
        self.cr.close_path()
        self.cr.fill()

    def draw_compass(self):
        '''draw a compass'''
        # out circle
        self.cr.set_source_rgba(1, 1, 1, 0.7)
        self.cr.set_line_width(3)
        self.cr.arc(WIN_X - 60, 60, CMPS_SIZE, 0, 2 * pi)
        self.cr.stroke()

        # position of compass pointer
        nloc = cmath.rect(CMPS_SIZE, self.rotate + pi / 2)
        x, y = nloc.real + WIN_X - 60, nloc.imag + WIN_Y - 60
        self.cr.arc(x, WIN_Y - y, CMPS_N_SIZE + 3, 0, 2 * pi)
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
        self.cr.move_to(shape_n[0].real, WIN_Y - shape_n[0].imag)
        for point in shape_n[1:]:
            self.cr.line_to(point.real, WIN_Y - point.imag)
        self.cr.stroke()

    def get_x_y(self):
        rect = self.get_allocation()
        x = (rect.x + rect.width / 2.0)
        y = (rect.y + rect.height / 2.0) - 20
        return x, y

    def _mouseScroll(self, widget, event):
        zoomfactor = 1.2 if (event.direction == gdk.SCROLL_UP) else 0.8
        k_old = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        self.zoomlevel *= zoomfactor
        if self.zoomlevel < MAX_ZOOMLEVEL:
            self.zoomlevel = MAX_ZOOMLEVEL

        # center of mouse
        cm = complex(float(event.x), WIN_Y - float(event.y))
        k_new = cmath.rect(self.zoomlevel * SCALE, self.rotate)
        ref_new = cm * (k_new - k_old) / (k_new * k_old) + self.ref
        self.ref = ref_new

        self.timing['mouse_zoom'] = time.time()
        # delayed refresh
        t_from_lastzoom = time.time() - self.timing['last_zoom']
        if t_from_lastzoom > DELAY_UPDATE:
            self.refresh_shoreline()
            self.queue_draw()
            self.timing['last_zoom'] = time.time()
            self.timing['update_waiting'] = False
        else:
            self.timing['update_waiting'] = True

    def _mouse_press(self, widget, event):
        self._mouseX, self._mouseY = event.x, event.y

    def _mouse_release(self, widget, event):
        '''rotate if mouse pressed in compass N area, otherwise pan move'''
        cm = complex(self._mouseX, WIN_Y - self._mouseY)
        self._dragDeltaX = event.x - self._mouseX
        self._dragDeltaY = event.y - self._mouseY
        # position of compass pointer
        nloc = cmath.rect(CMPS_SIZE, self.rotate + pi / 2)
        cc = complex(WIN_X - 60, WIN_Y - 60)  # center of big compass circle
        cn = nloc + complex(WIN_X - 60, WIN_Y - 60)  # center of N
        # mouse start from little N
        if abs(cm - cn) <= (CMPS_N_SIZE + 3) and not self.dialog:
            cnew = complex(event.x, WIN_Y - event.y)
            k_old = cmath.rect(self.zoomlevel * SCALE, self.rotate)
            # center of screen
            cscr_c = complex(WIN_X / 2, WIN_Y / 2)
            rotate_new = cmath.phase(cnew - cc) - pi / 2
            k_new = cmath.rect(self.zoomlevel * SCALE, rotate_new)
            c_new = cscr_c * (k_new - k_old) / (k_new * k_old) + self.ref
            self.ref = c_new
            self.rotate = rotate_new
        elif self.dialog and self._dragDeltaX == self._dragDeltaY == 0:
            if not self.flag_ruler_start:
                self.ruler_start = self.xy2wgs84(event.x, event.y)
                self.flag_ruler_start = True
                self.flag_ruler_end = False
            else:
                self.flag_ruler_end = True
                self.flag_ruler_start = False
                self.queue_draw()
        else:
            k = cmath.rect(self.zoomlevel * SCALE, self.rotate)
            self.ref -= complex(self._dragDeltaX, -1 * self._dragDeltaY) / k

        self.refresh_shoreline()
        self.queue_draw()

    def _mouse_move(self, widget, event):
        if self.dialog and self.flag_ruler_start and not self.flag_ruler_end:
            ruler_end = self.xy2wgs84(event.x, event.y)
            self.ruler_distance = earthdistance(ruler_end, self.ruler_start)
            self.label_12.set_text('{0:.2f}'.format(self.ruler_distance *
                                                    self.unitfactor))
            ruler_heading = cmath.phase(ruler_end - self.ruler_start)
            self.label_22.set_text('{0:.2f}'.format(
                                    (90 - ruler_heading * 180 / pi) % 360))

        self.pointer_x, self.pointer_y = event.x, event.y

        t_from_lastzoom = time.time() - self.timing['last_zoom']
        if t_from_lastzoom > DELAY_UPDATE:
            self.timing['update_waiting'] = False

        if not self.timing['update_waiting']:
            self.queue_draw()


class Main(object):
    def __init__(self, host='localhost', port='2947', device=None, debug=0):
        self.host = host
        self.port = port
        self.device = device
        self.debug = debug
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_size_request(WIN_X, WIN_Y)
        if not self.window.get_display():
            raise Exception("Can't open display")
        self.window.set_title('Navigator')
        self.widget = Navigation()
        self.window.connect('delete_event', self.delete_event)
        self.window.connect('destroy', self.destroy)
        self.widget.show()
        vbox = gtk.VBox(False, 0)
        self.window.add(vbox)
        self.window.present()

        buttonbox = gtk.HButtonBox()
        buttonbox.set_layout(gtk.BUTTONBOX_END)
        button_toggle_placemark = gtk.ToggleButton('Placemark')
        button_toggle_placemark.connect('toggled', self.toggle_placemark)
        buttonbox.pack_start(button_toggle_placemark, expand=True, fill=True)

        button_dist = gtk.Button('ruler')
        button_dist.connect('clicked', self.distance_dialog)
        buttonbox.pack_start(button_dist, expand=True, fill=True)

        button_reset_rotate = gtk.Button('Reset Rotate')
        button_reset_rotate.connect('clicked', self.reset_rotate)
        buttonbox.pack_start(button_reset_rotate, expand=True, fill=True)

        vbox.pack_start(buttonbox, False, False, 0)
        vbox.add(self.widget)
        self.window.show_all()
        self.ruler_start = False
        self.dialog = False

    def distance_dialog(self, widget):
        '''a dialog like google earth distance tool'''
        self.widget.dialog = True
        self.message = gtk.Dialog(title='ruler')
        self.message.connect('destroy', self.toggle_dialog)
        table = gtk.Table(rows=4, columns=3, homogeneous=True)
        table.set_row_spacings(5)
        table.set_col_spacings(5)
        self.combo = gtk.combo_box_new_text()
        self.combo.append_text('Meters')
        self.combo.append_text('Kilometers')
        self.combo.append_text('Nautical Miles')
        self.combo.append_text('Miles')
        self.combo.set_active(0)
        table.attach(self.combo,
                     left_attach=2,
                     right_attach=3,
                     top_attach=1,
                     bottom_attach=2)

        self.label_00 = gtk.Label('Measure the distance between two points'
                                  ' on the ground')

        table.attach(self.label_00, xoptions=gtk.FILL,
                     left_attach=0, right_attach=3,
                     top_attach=0, bottom_attach=1)

        self.label_11 = gtk.Label('Map Length:')
        self.label_11.set_justify(gtk.JUSTIFY_RIGHT)
        table.attach(self.label_11, xoptions=gtk.FILL,
                     left_attach=0, right_attach=1,
                     top_attach=1, bottom_attach=2)

        self.label_21 = gtk.Label('Heading:')
        self.label_21.set_justify(gtk.JUSTIFY_RIGHT)
        table.attach(self.label_21, xoptions=gtk.FILL,
                     left_attach=0, right_attach=1,
                     top_attach=2, bottom_attach=3)

        self.label_23 = gtk.Label('degrees')
        self.label_23.set_justify(gtk.JUSTIFY_LEFT)
        table.attach(self.label_23, xoptions=gtk.FILL,
                     left_attach=2, right_attach=3,
                     top_attach=2, bottom_attach=3)

        self.widget.label_12 = gtk.Label('')
        self.widget.label_12.set_justify(gtk.JUSTIFY_RIGHT)
        table.attach(self.widget.label_12,
                     left_attach=1, right_attach=2,
                     top_attach=1, bottom_attach=2)

        self.widget.label_22 = gtk.Label('')
        self.widget.label_22.set_justify(gtk.JUSTIFY_RIGHT)
        table.attach(self.widget.label_22,
                     left_attach=1, right_attach=2,
                     top_attach=2, bottom_attach=3)

        self.message.vbox.pack_start(table)
        table.show()
        self.combo.connect('changed', self.change_unit)
        self.combo.show()
        self.label_00.show()
        self.label_11.show()
        self.label_21.show()
        self.label_23.show()
        self.widget.label_12.show()
        self.widget.label_22.show()
        self.message.show()

    def change_unit(self, widget):
        model = self.combo.get_model()
        index = self.combo.get_active()
        unit = model[index][0]
        self.widget.unitfactor = UNIT[unit]
        self.widget.label_12.set_text('{0:.2f}'.format(
                        self.widget.ruler_distance * self.widget.unitfactor))

    def toggle_dialog(self, widget):
        self.widget.dialog = False
        self.widget.flag_ruler_start = False
        self.widget.flag_ruler_end = False

    def watch(self, daemon, device):
        self.daemon = daemon
        self.device = device
        gobject.io_add_watch(daemon.sock, gobject.IO_IN, self.handle_response)
        gobject.io_add_watch(daemon.sock, gobject.IO_ERR, self.handle_hangup)
        gobject.io_add_watch(daemon.sock, gobject.IO_HUP, self.handle_hangup)
        return True

    def handle_response(self, source, condition):
        if self.daemon.read() == -1:
            self.handle_hangup(source, condition)
        if self.daemon.data['class'] == 'TPV':
            self.update_speed(self.daemon.data)
        if self.daemon.data['class'] == 'SKY':
            self.update_skyview(self.daemon.data)

        return True

    def handle_hangup(self, dummy, unused):
        w = gtk.MessageDialog(
                type=gtk.MESSAGE_ERROR,
                flags=gtk.DIALOG_DESTROY_WITH_PARENT,
                buttons=gtk.BUTTONS_OK
        )
        w.connect("destroy", lambda unused: gtk.main_quit())
        w.set_title('gpsd error')
        w.set_markup("gpsd has stopped sending data.")
        w.run()
        gtk.main_quit()
        return True

    def rotate_right(self, widget):
        self.rotate(widget, -2)

    def rotate_left(self, widget):
        self.rotate(widget, 2)

    def reset_rotate(self, widget):
        self.widget.rotate = 0

    def rotate(self, widget, angle):
        rotate_new = self.widget.rotate + angle * pi / 180
        k_old = cmath.rect(self.widget.zoomlevel * SCALE, self.widget.rotate)
        # center of screen
        cm = complex(WIN_X / 2, WIN_Y / 2)
        k_new = cmath.rect(self.widget.zoomlevel * SCALE, rotate_new)
        c_new = cm * (k_new - k_old) / (k_new * k_old) + self.widget.ref
        self.widget.ref = c_new
        self.widget.rotate = rotate_new
        self.widget.queue_draw()

    def toggle_placemark(self, widget):
        if widget.get_active():
            self.widget.show_placemarks = True
        else:
            self.widget.show_placemarks = False

    def update_speed(self, data):
        if hasattr(data, 'time'):
            tstr = [c for c in data.time[:-5]]
            tstr[10] = ' '
            tstr.append(' UTC')
            self.widget.utc_time = ''.join(tstr)

        if hasattr(data, 'speed'):
            self.widget.last_speed = data.speed
            #self.widget.queue_draw()
        if hasattr(data, 'track'):
            self.widget.heading = (90 - int(data.track)) * pi / 180
            self.widget.heading_degree = int(data.track)
            #self.widget.queue_draw()
        if hasattr(data, 'lon') and hasattr(data, 'lat'):
            pos = complex(float(data.lon), float(data.lat))
            distance = earthdistance(pos, self.widget.position)
            #print '\ndistance between points %f m' % distance
            self.widget.track_refresh_cnt += distance
            if self.widget.track_refresh_cnt > 5:
                self.widget.queue_draw()
                self.widget.track_refresh_cnt = 0
            #if distance > 50:
            self.widget.position = pos
            self.widget.track[self.widget.track_indx] = pos
            self.widget.track_indx += 1

    def update_skyview(self, data):
        "Update the satellite list and skyview."
        if hasattr(data, 'satellites'):
            self.widget.satellites = data.satellites
            #self.widget.queue_draw()

    def delete_event(self, widget, event, data=None):
        #TODO handle all cleanup operations here
        return False

    def destroy(self, unused, empty=None):
        gtk.main_quit()

    def run(self):
        try:
            daemon = gps.gps(
                host=self.host,
                port=self.port,
                mode=gps.WATCH_ENABLE | gps.WATCH_JSON | gps.WATCH_SCALED,
                verbose=self.debug
            )
            self.watch(daemon, self.device)
            gtk.main()
        except SocketError:
            w = gtk.MessageDialog(
                    type=gtk.MESSAGE_ERROR,
                    flags=gtk.DIALOG_DESTROY_WITH_PARENT,
                    buttons=gtk.BUTTONS_OK
            )
            w.set_title('socket error')
            w.set_markup(
                "could not connect to gpsd socket. make sure gpsd is running."
            )
            w.run()
            w.destroy()
        except KeyboardInterrupt:
            self.window.emit('delete_event', gtk.gdk.Event(gtk.gdk.NOTHING))


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
