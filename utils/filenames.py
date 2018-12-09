"""Naming conventions for geospatial data files."""

import datetime
import logging
import os
import re
import tarfile
import zipfile


# https://landsat.usgs.gov/landsat-collections
# https://landsat.usgs.gov/what-are-naming-conventions-landsat-scene-identifiers
LANDSAT_SENSOR = {
    'C': 'OLI/TIRS Combined',
    'O': 'OLI-only',
    # 'T': 'TIRS-only',  # duplicated dict entry
    'E': 'ETM+',
    'T': 'TM',
    'M': 'MSS',
}

LANDSAT_PROCESSING_CORRECTION_LEVEL = {
    'L1TP': 'Precision Terrain',
    'L1GT': 'Systematic Terrain',
    'L1GS': 'Systematic',
}

LANDSAT_COLLECTION_CATEGORY = {
    'RT': 'Real-Time',
    'T1': 'Tier 1',
    'T2': 'Tier 2',
}

# https://landsat.usgs.gov/what-are-band-designations-landsat-satellites
LANDSAT_BAND_NAME = {
    5: {
        1: 'Blue',
        2: 'Green',
        3: 'Red',
        4: 'NIR',
        5: 'SWIR-1',
        6: 'Thermal',
        7: 'SWIR-2',
        'Blue': 1,
        'Green': 2,
        'Red': 3,
        'NIR': 4,
        'SWIR-1': 5,
        'Thermal': 6,
        'SWIR-2': 7,
    },
    7: {
        1: 'Blue',
        2: 'Green',
        3: 'Red',
        4: 'NIR',
        5: 'SWIR-1',
        6: 'Thermal',
        7: 'SWIR-2',
        8: 'Panchromatic',
        'Blue': 1,
        'Green': 2,
        'Red': 3,
        'NIR': 4,
        'SWIR-1': 5,
        'Thermal': 6,
        'SWIR-2': 7,
        'Panchromatic': 8,
    },
    8: {
        1: 'Ultra Blue',
        2: 'Blue',
        3: 'Green',
        4: 'Red',
        5: 'NIR',
        6: 'SWIR-1',
        7: 'SWIR-2',
        8: 'Panchromatic',
        9: 'Cirrus',
        10: 'TIRS-1',
        11: 'TIRS-2',
        'Ultra Blue': 1,
        'Blue': 2,
        'Green': 3,
        'Red': 4,
        'NIR': 5,
        'SWIR-1': 6,
        'SWIR-2': 7,
        'Panchromatic': 8,
        'Cirrus': 9,
        'TIRS-1': 10,
        'TIRS-2': 11,
    },
}

# The spectral bands we want to collect data for
BANDS_OF_INTEREST = [
    'Blue',
    'Green',
    'Red',
    'NIR',
    'SWIR-1',
    'SWIR-2',
]

# https://landsat.usgs.gov/landsat-surface-reflectance-quality-assessment
LANDSAT_PIXEL_QA_NAME = {
    # Landsat 5, 7
    1: 'Fill',
    66: 'Clear, low-confidence cloud',
    68: 'Water, low-confidence cloud',
    72: 'Cloud shadow, low-confidence cloud',
    80: 'Snow/ice, low-confidence cloud',
    96: 'Cloud, low-confidence cloud',
    112: 'Snow/ice, cloud, low-confidence cloud',
    130: 'Clear, medium-confidence cloud',
    132: 'Water, medium-confidence cloud',
    136: 'Cloud shadow, medium-confidence cloud',
    144: 'Snow/ice, medium-confidence cloud',
    160: 'Cloud, medium-confidence cloud',
    176: 'Snow/ice, cloud, medium-confidence cloud',
    224: 'Cloud, high-confidence cloud',
    # Landsat 8
    322: 'Clear terrain, low confidence cloud, low confidence cirrus',
    324: 'Water, low confidence cloud, low confidence cirrus',
    328: 'Cloud shadow, low confidence cloud, low confidence cirrus',
    336: 'Snow/Ice, low confidence cloud, low confidence cirrus',
    352: 'Cloud, low confidence cloud, low confidence cirrus',
    368: 'Snow/Ice, cloud, low confidence cloud, low confidence cirrus',
    386: 'Clear terrain, medium confidence cloud, low confidence cirrus',
    388: 'Water, medium confidence cloud, low confidence cirrus',
    392: 'Cloud shadow, medium confidence cloud, low confidence cirrus',
    400: 'Snow/Ice, medium confidence cloud, low confidence cirrus',
    416: 'Cloud, medium confidence cloud, low confidence cirrus',
    432: 'Snow/Ice, cloud, medium confidence cloud, low confidence cirrus',
    480: 'Cloud, high confidence cloud, low confidence cirrus',
    834: 'Clear terrain, low confidence cloud, high confidence cirrus',
    836: 'Water, low confidence cloud, high confidence cirrus',
    840: 'Cloud shadow, low confidence cloud, high confidence cirrus',
    848: 'Snow/Ice, low confidence cloud, high confidence cirrus',
    864: 'Cloud, low confidence cloud, high confidence cirrus',
    880: 'Snow/Ice, cloud, low confidence cloud, high confidence cirrus',
    898: 'Clear terrain, medium confidence cloud, high confidence cirrus',
    900: 'Water, medium confidence cloud, high confidence cirrus',
    904: 'Cloud shadow, medium confidence cloud, high confidence cirrus',
    912: 'Snow/Ice, medium confidence cloud, high confidence cirrus',
    928: 'Cloud, medium confidence cloud, high confidence cirrus',
    944: 'Snow/Ice, cloud, medium confidence cloud, high confidence cirrus',
    992: 'Cloud, high confidence cloud, high confidence cirrus',
    1346: 'Clear terrain, terrain occluded',
    1348: 'Water, terrain occluded',
    1350: 'Cloud shadow, terrain occluded',
    1352: 'Snow/ice, terrain occluded',
}

# The pixel QA values denoting clear sky
PIXEL_QA_OF_INTEREST = [
    # Landsat 5, 7
    66,
    # Landsat 8
    322,
]

LANDSAT_BAND_RE = re.compile('^b(\d+)$')
LANDSAT_SR_BAND_RE = re.compile('^sr_band(\d+)$')

DATE_STR = '%Y%m%d'
DATETIME_STR = '%Y%m%d%H%M%S'


class CroplandDataLayer:
    """Example:

    * CDL_2010_clip_20180517162110_1602234420.zip
    """

    def __init__(self, filename):
        # File
        self.filename = filename
        base = os.path.basename(filename)
        stem = os.path.splitext(base)[0]

        # Dates
        self.acquisition_year = int(stem[4:8])
        self.processing_date = datetime.datetime.strptime(
            stem[14:28], DATETIME_STR)

        # Final number in filename is a random hash, ignore it

        # Attribute to use later
        self._names = None

    def __str__(self):
        """String version of an object.

        Returns:
            str: string version of an object
        """
        string = """\
Filename: {self.filename}
Acquisition Year: {self.acquisition_year}
Processing Date: {self.processing_date}"""

        return string.format(self=self)

    def __repr__(self):
        """Printable representation of an object.

        Returns:
            str: printable representation of an object
        """
        string = "{self.__class__.__name__}('{self.filename}')"
        return string.format(self=self)

    def __lt__(self, other):
        """Less than.

        By defining this operator, we can sort a list of
        CroplandDataLayer objects.

        Returns:
            bool: True if self < other, else False
        """
        return self.acquisition_year < other.acquisition_year

    def __eq__(self, other):
        """Equals.

        By defining this operator, we can use itertools.groupby to group
        a list of CroplandDataLayer objects by acquisition year.

        Returns:
            bool: True if self == other, else False
        """
        return self.acquisition_year == other.acquisition_year

    def __hash__(self):
        """Hashing function.

        By defining this special function, we allow CroplandDataLayer
        objects to be cached.

        Returns:
            int: a unique hash identifying this instance
        """
        return hash(self.filename)

    @property
    def names(self):
        """A list of filenames contained in the zip file.

        Peeks inside of the zip file to get a list of files contained within.
        Obviously this operation is slow, so the results are cached.

        Returns:
            list: a list of filenames
        """
        if self._names:
            return self._names

        logging.info('Peeking inside ' + self.filename)

        self._names = zipfile.ZipFile(self.filename).namelist()

        return self._names

    def tif(self):
        """The .tif filename.

        Returns:
            str: the .tif filename
        """
        for name in self.names:
            if name.endswith('.tif'):
                # Check if file has already been decompressed
                dirname = os.path.dirname(self.filename)
                path = os.path.join(dirname, name)
                if os.path.exists(path):
                    return path
                return '/vsizip/' + os.path.join(self.filename, name)

    def dbf(self):
        """The .dbf filename.

        Returns:
            str: the .dbf filename
        """
        for name in self.names:
            if name.endswith('.dbf'):
                # Check if file has already been decompressed
                dirname = os.path.dirname(self.filename)
                path = os.path.join(dirname, name)
                if os.path.exists(path):
                    return path
                return '/vsizip/' + os.path.join(self.filename, name)


class LandsatOrderIdentifier:
    """Examples:

    * LT050310312010110701T1-SC20180530134137.tar.gz
    * LE070310312010072601T1-SC20180608102512.tar.gz
    * LC080310322017022701T1-SC20180704175637.tar.gz
    """

    def __init__(self, filename):
        # File
        self.filename = filename
        base = os.path.basename(filename)
        stem = os.path.splitext(base)[0]

        # Satellite
        self.sensor = stem[1]
        self.satellite = int(stem[2:4])
        self.wrs_path = int(stem[4:7])
        self.wrs_row = int(stem[7:10])

        # Dates
        self.acquisition_date = datetime.datetime.strptime(
            stem[10:18], DATE_STR)
        self.processing_date = datetime.datetime.strptime(
            stem[25:39], DATETIME_STR)

        # Collection
        self.collection_number = int(stem[18:20])
        self.collection_category = stem[20:22]

        # Attribute to use later
        self._names = None

    def __str__(self):
        """String version of an object.

        Returns:
            str: string version of an object
        """
        string = """\
Filename: {self.filename}
Sensor: {sensor}
Satellite: Landsat {self.satellite}
WRS Path: {self.wrs_path}
WRS Row: {self.wrs_row}
Acquisition Date: {self.acquisition_date}
Processing Date: {self.processing_date}
Collection Number: {self.collection_number}
Collection Category: {collection_category}"""

        return string.format(
            self=self,
            sensor=LANDSAT_SENSOR[self.sensor],
            collection_category=LANDSAT_COLLECTION_CATEGORY[
                self.collection_category
            ])

    def __repr__(self):
        """Printable representation of an object.

        Returns:
            str: printable representation of an object
        """
        string = "{self.__class__.__name__}('{self.filename}')"
        return string.format(self=self)

    def __lt__(self, other):
        """Less than.

        By defining this operator, we can sort a list of
        LandsatOrderIdentifier objects.

        Returns:
            bool: True if self < other, else False
        """
        return self.acquisition_date < other.acquisition_date

    def __eq__(self, other):
        """Equals.

        By defining this operator, we can use itertools.groupby to group
        a list of LandsatOrderIdentifier objects by acquisition date.

        Returns:
            bool: True if self == other, else False
        """
        return self.acquisition_date == other.acquisition_date

    def __hash__(self):
        """Hashing function.

        By defining this special function, we allow LandsatOrderIdentifier
        objects to be cached.

        Returns:
            int: a unique hash identifying this instance
        """
        return hash(self.filename)

    @property
    def names(self):
        """A list of filenames contained in the tar file.

        Peeks inside of the tar file to get a list of files contained within.
        Obviously this operation is slow, so the results are cached.

        Returns:
            list: a list of LandsatProductIdentifier objects
        """
        if self._names:
            return self._names

        logging.info('Peeking inside ' + self.filename)

        with tarfile.open(self.filename) as tar:
            names = tar.getnames()

        self._names = list(map(LandsatProductIdentifier, names))

        return self._names

    def pixel_qa(self):
        """The pixel QA filename.

        Returns:
            str: the pixel QA filename
        """
        for name in self.names:
            if name.product == 'pixel_qa':
                # Check if file has already been decompressed
                dirname = os.path.dirname(self.filename)
                path = os.path.join(dirname, name.filename)
                if os.path.exists(path):
                    return path
                return '/vsitar/' + os.path.join(self.filename, name.filename)

    def sr_band(self, band):
        """The sr band filename.

        Parameters:
            band (str): the name of the band

        Returns:
            str: the sr band filename
        """
        sr_band = 'sr_band{}'.format(LANDSAT_BAND_NAME[self.satellite][band])
        for name in self.names:
            if name.product == sr_band:
                # Check if file has already been decompressed
                dirname = os.path.dirname(self.filename)
                path = os.path.join(dirname, name.filename)
                if os.path.exists(path):
                    return path
                return '/vsitar/' + os.path.join(self.filename, name.filename)


class LandsatProductIdentifier:
    """Examples:

    * LT05_L1TP_031031_20101107_20160831_01_T1.xml
    * LE07_L1TP_031031_20100726_20160915_01_T1.xml
    * LC08_L1TP_031032_20170227_20170316_01_T1.xml
    """

    def __init__(self, filename):
        # File
        self.filename = filename
        base = os.path.basename(filename)
        stem = os.path.splitext(base)[0]

        # Satellite
        self.sensor = stem[1]
        self.satellite = int(stem[2:4])
        self.processing_correction_level = stem[5:9]
        self.wrs_path = int(stem[10:13])
        self.wrs_row = int(stem[13:16])

        # Dates
        self.acquisition_date = datetime.datetime.strptime(
            stem[17:25], DATE_STR)
        self.processing_date = datetime.datetime.strptime(
            stem[26:34], DATE_STR)
        self.doy = self.acquisition_date.timetuple().tm_yday

        # Collection
        self.collection_number = int(stem[35:37])
        self.collection_category = stem[38:40]

        # Product
        self.product = None
        if len(stem) > 40:
            self.product = stem[41:]

        # Band
        self.band = None
        if self.product:
            for regex in [LANDSAT_SR_BAND_RE, LANDSAT_BAND_RE]:
                match = regex.fullmatch(self.product)
                if match:
                    self.band = int(match.group(1))
                    break

    def __str__(self):
        """String version of an object.

        Returns:
            str: string version of an object
        """
        string = """\
Filename: {self.filename}
Sensor: {sensor}
Satellite: Landsat {self.satellite}
Processing Correction Level: {pcl}
WRS Path: {self.wrs_path}
WRS Row: {self.wrs_row}
Acquisition Date: {self.acquisition_date}
Processing Date: {self.processing_date}
Collection Number: {self.collection_number}
Collection Category: {collection_category}"""

        if self.product:
            string += """
Product: {self.product}"""

        if self.band:
            string += """
Band: {band}""".format(band=LANDSAT_BAND_NAME[self.satellite][self.band])

        return string.format(
            self=self,
            sensor=LANDSAT_SENSOR[self.sensor],
            pcl=LANDSAT_PROCESSING_CORRECTION_LEVEL[
                self.processing_correction_level
            ],
            collection_category=LANDSAT_COLLECTION_CATEGORY[
                self.collection_category
            ])

    def __repr__(self):
        """Printable representation of an object.

        Returns:
            str: printable representation of an object
        """
        string = "{self.__class__.__name__}('{self.filename}')"
        return string.format(self=self)

    def __lt__(self, other):
        """Less than.

        By defining this operator, we can sort a list of
        LandsatProductIdentifier objects.

        Returns:
            bool: True if self < other, else False
        """
        return self.acquisition_date < other.acquisition_date

    def __eq__(self, other):
        """Equals.

        By defining this operator, we can use itertools.groupby to group
        a list of LandsatProductIdentifier objects by acquisition date.

        Returns:
            bool: True if self == other, else False
        """
        return self.acquisition_date == other.acquisition_date

    def __hash__(self):
        """Hashing function.

        By defining this special function, we allow LandsatProductIdentifier
        objects to be cached.

        Returns:
            int: a unique hash identifying this instance
        """
        return hash(self.filename)
