# from tethys_wrf.main import WRF
# from tethys_wrf import sio
from functools import wraps
import pyproj

# Default proj
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
