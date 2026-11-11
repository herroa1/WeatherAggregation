"""Settings for how the website totals rain and blends sources. Change them here; the next build uses them."""

# Rain is shown as totals over the RAIN_HOURS before each hour
RAIN_HOURS = 3

# Sources with steps more than an hour apart (e.g. ECMWF AIFS and NOAA AIGFS, 6-hourly) have each step's
# rain shared out over its hours like the other sources' mean hourly rain in those hours (see
# `website.blend.rain_profile`), instead of evenly. RAIN_SPLIT_SMOOTHING_MM (mm) is added to each hour of
# that mean before sharing, so a near-dry hint (0.1 mm against 0.0 mm) doesn't put all the rain in one
# hour: the bigger it is, the closer to an even split. Where no other source has hourly rain, it's even
RAIN_SPLIT_SMOOTHING_MM = 0.25
# Grid (degrees) the other sources' hourly rain is averaged onto to make that profile
RAIN_PROFILE_RES = 0.5
