#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''parse GEODAS 2-D Vector Data Format VCT00.

VCT00 is the 2-D vector data format by NOAA. It can be either binary or
ASCII. This script trying to parse its binary format and convert it to a
sqlite database file.

Get the world vector shoreline from:
http://www.ngdc.noaa.gov/mgg/dat/geodas/coastlines/LittleEndian/wvs1mres.b00
wvs1mres.b00: World Vector Shoreline subset at 1:1million resolution

the structure of output sqlite database is:
    lon REAL  => longitude
    lat REAL  => latitude
    segment INTEGER => which segment the point belongs to
    penstart INTEGER => pencode, 1 for start a new line
    res1 INTEGER   => full resolution
    res2 INTEGER   => high resolution
    res3 INTEGER   => medium-high resolution
    res4 INTEGER   => medium resolution
    res5 INTEGER   => medium-low resolution
    res6 INTEGER   => low resolution
    res7 INTEGER   => crude resolution

see vct00.pdf under the gridid_data directory for detail.

data structure of VCT00 binary format:

quote from the GEODAS document vct00.pdf

" Each VCT00 Header is 40 bytes in length and consists of 4 10-byte records,
each of which contains two signed 4-byte integers followed by one signed 2 byte
integer. The Header records are all together at the start of the file. The last
Header Record is an “empty” Header. This indicates that there are no more
Headers. Data records immediately follow this Header."

......


the data part:
    10 byte per record, each record contains:
    int latitude; 4 byte
    int longitude; 4 byte
    int pencode;  2 byte

'''

__license__   = 'BSD'
__copyright__ = '2013, Chen Wei <weichen302@gmx.com>'

from struct import unpack
import argparse
import os
import sqlite3

RESOLUTION_FACTOR = (1, 2, 3, 5, 7, 11, 13)


def initdb():
    print 'Removing old sqlite file'
    try:
        os.remove(DB_FILE)
    except OSError:
        pass
    conn = sqlite3.connect(DB_FILE)
    db = conn.cursor()
    db.execute('''CREATE TABLE IF NOT EXISTS shore
                   (id INTEGER PRIMARY KEY,
                    lon REAL,
                    lat REAL,
                    penstart INTEGER,
                    segment INTEGER,
                    res1 INTEGER,
                    res2 INTEGER,
                    res3 INTEGER,
                    res4 INTEGER,
                    res5 INTEGER,
                    res6 INTEGER,
                    res7 INTEGER)''')


def pencode2res(pencode):
    '''convert VCT00 pencode to cooresponde resolution Level
    Pencode          Resolution
    -------          ----------
    1                level 1
    factor of 2      level 2
    factor of 3      level 3
    factor of 5      level 4
    factor of 7      level 5
    factor of 11     level 6
    factor of 13     level 7
    Args:
        pencode: a integer
    Return:
        a list of resolutions, eg. [0, 0, 1, 0, 1, 0, 0]
        1 means this point is included in this resolution level
    '''
    res = [0, 0, 0, 0, 0, 0, 0]
    for i in xrange(7):
        if pencode % RESOLUTION_FACTOR[i] == 0:
            res[i] = 1
    return res


def unpack_vct00(tenbytes):
    '''unpack a 10-bytes vct00 record section
    all sections are seperated into:
    4-bytes, 4-bytes, 2-bytes, all signed integers
    Args:
        tenbytes: the 10 bytes section
    Return:
        a list of 3 integer, eg. [1, 22, 333]
        '''
    lat = unpack('<i', tenbytes[:4])[0]    # lat  long integer
    lon = unpack('<i', tenbytes[4:8])[0]   # lon
    pen = unpack('<h', tenbytes[8:])[0]  # pencode
    return (lat, lon, pen)


def parse_vct00_header(fname):
    '''each VCT00 header is 40 bytes long, consists 4 10-bytes long data, all
    signed integers.

               4 bytes                   4 bytes         2 bytes
               -------                   -------         -------
1st 10-byte:   nth 10byte,first point    number of       type of vector
               of vector data            points
2nd 10-byte:   type                      N/U             N/U
3rd 10-byte:   NW-Latitude               NW-Longitude    N/U
4th 10-byte:   SE-Latitude               SE-Longitude    N/U

the NOAA document vct00.pdf is not exactly accurate on how header section ends.
The end of header appears has type of -1 in the first 10 bytes of the 40 bytes
block, and not all 40 bytes are "empty", only 30-40 bytes are empty.

    Args:
        fname: the file name of VCT00 file
    Return:
        a list of tuple contains segment start position in VCT00 file and
        total points in that segment
        [(position, point count), ...]
    '''
    print 'Reading %s' % fname
    datafp = open(fname, 'rb').read()
    start = 0
    headers = []  # list of (record position, number of points in segment)
    while start < len(datafp):
        header = []
        sec = datafp[start: start + 40]
        if len(sec) < 40:
            print len(sec)
        start += 40
        sec_start = 0
        while sec_start < 40:
            line = sec[sec_start: sec_start + 10]
            if len(line) != 10:
                break
            tmp = unpack_vct00(line)
            header.append(tmp)
            sec_start += 10

        if header[0][2] == -1:
            # end of headers
            break
        else:
            headers.append(header[0])
    print 'total %d segment' % len(headers)
    return headers


def parse_vct00_data(fname):
    print 'Reading %s' % fname
    headers = parse_vct00_header(fname)
    datafp = open(fname, 'rb').read()
    start = 0
    res = []
    segid = 0

    for start, ptcount, d_type in headers:
        start -= 1  # record count starts at 1 in VCT00
        segid += 1
        resolutions = [1] * 7
        sec = datafp[start * 10: (start + ptcount) * 10]
        sec_start = 0
        penstart = 1
        while sec_start < len(sec):
            line = sec[sec_start: sec_start + 10]
            sec_start += 10
            if len(line) != 10:
                break
            lat, lon, pen = unpack_vct00(line)

            pt = [lon / 1000000.0, lat / 1000000.0, segid]
            if pen == 0:
                penstart = 1
                resolutions = [1] * 7
            else:
                penstart = 0
                resolutions = pencode2res(pen)
            pt.append(penstart)
            pt.extend(resolutions)
            res.append(pt)

    return res


def writedb(pts, dbfp):
    '''write records to database'''
    rcdcount = {'segment': 0, 'resolution': [0] * 7}
    conn = sqlite3.connect(dbfp)
    db = conn.cursor()
    writecount = 0
    total = len(pts)
    for rcd in pts:
        db.execute('''INSERT INTO shore
               (lon, lat, segment, penstart,
               res1, res2, res3, res4, res5, res6, res7)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)''', rcd)

        writecount += 1
        if writecount % 90000 == 0:
            print '{0:.0%}, {1} points wrote to sqlite, {2} left'.format(
                float(writecount) / total, writecount, total - writecount)
        if rcd[3] == 1:
            rcdcount['segment'] += 1
        for i in xrange(7):
            if rcd[4 + i] == 1:
                rcdcount['resolution'][i] += 1

    conn.commit()
    db.close()

    return rcdcount


def main():
    initdb()
    rcds = parse_vct00_data(COAST_FILE)
    rcdcount = writedb(rcds, DB_FILE)
    print 'Total %d points' % len(rcds)
    print 'Total %d pen draw segment' % rcdcount['segment']
    for i in xrange(7):
        print '   - resolution %d: %d points' % (i + 1,
                                                 rcdcount['resolution'][i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path of VCT00 file')
    parser.add_argument('-o', '--output', help='path of output SQLITE file')
    args = parser.parse_args()
    DB_FILE = args.output
    COAST_FILE = args.input

    main()
