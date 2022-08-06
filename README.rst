tethys-wrf
==========

WRF variable requirements
----------------------
The base requirements are the XLONG and XLAT dimensions as well as the Times variable. The height_agl listed below is required for any non-surface variables.

height_agl::

  'PH', 'PHB', 'HGT'

air_temperature::

  'T', 'P', 'PB', 'T2'

wind::

  'U', 'V', 'U10', 'V10'

relative_humidity::

  'T', 'P', 'PB', 'QVAPOR', 'T2', 'PSFC', 'Q2'

dew_temperature::

  'P', 'PB', 'QVAPOR', 'PSFC', 'Q2'

air_pressure::

  'P', 'PRES', 'PSFC'

precipitation::

  'RAINNC'

snowfall::

  'SNOWNC'

surface_runoff::

  'SFROFF'

gw_recharge::

  'UDROFF'

downward_shortwave::

  'SWDOWN'

downward_longwave::

  'GLW'

ground_heat_flux::

  'GRDFLX'

soil_temperature::

  'TSLB', 'ZS'

soil_water::

  'SMOIS', 'ZS'

pblh::

  'PBLH'

albedo::

  'ALBEDO'

surface_emissivity::

  'EMISS'

water_vapor_mixing_ratio::

  QVAPOR', 'Q2'
