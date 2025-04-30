import numpy as np
from datetime import datetime, timedelta
import pytz
try:
    import pvlib
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    print("pvlib not available, using default solar position calculation")


def compute_nighttime_mask(timestamps, lats, lons, solar_zenith_angle=None, sza_format='degrees',
                          sza_threshold=90.0, timezone='UTC', use_pvlib=False, return_float=False):
    """
    Compute nighttime mask using the correct local time for each site or directly from solar zenith angle.

    Args:
        timestamps: Either a dictionary of {site_idx -> array of local timestamps}
                   OR a single array of timestamps for all sites.
        lats: Array of latitude values (degrees).
        lons: Array of longitude values (degrees).
        solar_zenith_angle: Optional pre-computed solar zenith angle values (time, sites).
                           Format specified by sza_format: 'degrees', 'cosine', or 'scaled_degrees' (degrees * 100).
        sza_format: Format of solar_zenith_angle: 'degrees', 'cosine', or 'scaled_degrees'.
        sza_threshold: Solar zenith angle threshold (degrees) for nighttime (default: 90.0).
        timezone: Timezone of timestamps ('UTC' or 'local'; default: 'UTC').
        use_pvlib: Use pvlib for precise solar position calculation if available (default: False).
        return_float: Return mask as float32 (0.0/1.0) instead of bool (default: False).

    Returns:
        nighttime_mask: Array (n_times, n_sites) indicating nighttime (1.0/True) or daytime (0.0/False).
    """
    # Validate inputs
    if not isinstance(lats, (list, np.ndarray)) or not isinstance(lons, (list, np.ndarray)):
        raise ValueError("lats and lons must be lists or NumPy arrays")
    lats = np.array(lats)
    lons = np.array(lons)
    if len(lats) != len(lons):
        raise ValueError("lats and lons must have the same length")

    # Handle timestamp formats
    if isinstance(timestamps, dict):
        timestamps_dict = timestamps
        n_times = len(next(iter(timestamps_dict.values())))
        # Validate timestamp consistency
        if len(set(len(t) for t in timestamps_dict.values())) > 1:
            raise ValueError("Inconsistent timestamp lengths across sites")
        if len(timestamps_dict) != len(lats):
            raise ValueError("Number of sites in timestamps_dict does not match lats/lons")
    else:
        n_times = len(timestamps)
        timestamps_dict = {site_idx: timestamps for site_idx in range(len(lats))}

    n_sites = len(lats)
    nighttime_mask = np.zeros((n_times, n_sites), dtype=bool)

    # Use pvlib if requested and available
    if use_pvlib and PVLIB_AVAILABLE and solar_zenith_angle is None:
        print("Calculating nighttime mask using pvlib")
        for site_idx in range(n_sites):
            times = timestamps_dict[site_idx]
            if timezone == 'UTC':
                # Convert UTC to local time using longitude-based offset
                utc_offset_hours = lons[site_idx] / 15  # 15° per hour
                local_tz = pytz.FixedOffset(utc_offset_hours * 60)
                times = [t.replace(tzinfo=pytz.UTC).astimezone(local_tz) for t in times]
            location = pvlib.location.Location(lats[site_idx], lons[site_idx])
            solar_position = location.get_solarposition(times)
            nighttime_mask[:, site_idx] = (solar_position['zenith'] >= sza_threshold)
        return nighttime_mask.astype(np.float32) if return_float else nighttime_mask

    # Use provided solar zenith angle if available
    if solar_zenith_angle is not None:
        if len(solar_zenith_angle.shape) != 2 or solar_zenith_angle.shape != (n_times, n_sites):
            raise ValueError(f"solar_zenith_angle has unexpected shape {solar_zenith_angle.shape}, expected ({n_times}, {n_sites})")

        print(f"Using solar zenith angle data (format: {sza_format})")
        if sza_format == 'scaled_degrees':
            if solar_zenith_angle.dtype in [np.int16, np.uint16, np.int32, np.uint32]:
                solar_zenith_degrees = solar_zenith_angle.astype(float) / 100.0
                print(f"Scaled degrees to range {np.min(solar_zenith_degrees):.2f}-{np.max(solar_zenith_degrees):.2f}°")
            else:
                raise ValueError("scaled_degrees format requires integer dtype")
            nighttime_mask = (solar_zenith_degrees >= sza_threshold)
        elif sza_format == 'cosine':
            cos_zenith = solar_zenith_angle.astype(float)
            if not np.all((cos_zenith >= -1.0) & (cos_zenith <= 1.0)):
                raise ValueError("Cosine SZA values must be in [-1, 1]")
            cos_threshold = np.cos(np.deg2rad(sza_threshold))
            nighttime_mask = (cos_zenith <= cos_threshold)
            print(f"Using cosine of zenith angle, threshold at {cos_threshold:.4f}")
        else:  # 'degrees'
            solar_zenith_degrees = solar_zenith_angle.astype(float)
            print(f"Degrees range {np.min(solar_zenith_degrees):.2f}-{np.max(solar_zenith_degrees):.2f}°")
            nighttime_mask = (solar_zenith_degrees >= sza_threshold)
        return nighttime_mask.astype(np.float32) if return_float else nighttime_mask

    # Calculate nighttime mask from timestamps and coordinates
    print("Calculating nighttime mask from timestamps and coordinates")
    lat_rad = np.deg2rad(lats)[:, None]  # (n_sites, 1)
    # Assume timestamps are same across sites for vectorization
    sample_times = timestamps_dict[0]
    doys = np.array([t.timetuple().tm_yday for t in sample_times])  # (n_times,)
    hours = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in sample_times])  # (n_times,)

    if timezone == 'UTC':
        # Adjust hours based on longitude
        utc_offset_hours = lons / 15  # (n_sites,)
        hours = hours[:, None] + utc_offset_hours[None, :]  # (n_times, n_sites)
    else:
        hours = hours[:, None]  # (n_times, 1)

    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doys) / 365))  # (n_times,)
    decl_rad = np.deg2rad(decl)[:, None]  # (n_times, 1)
    ha = (hours - 12) * 15  # (n_times, n_sites)
    ha_rad = np.deg2rad(ha)  # (n_times, n_sites)

    cos_theta = (np.sin(lat_rad) * np.sin(decl_rad) +
                 np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))  # (n_times, n_sites)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    cos_threshold = np.cos(np.deg2rad(sza_threshold))
    nighttime_mask = (cos_theta <= cos_threshold)

    return nighttime_mask.T.astype(np.float32) if return_float else nighttime_mask.T


def compute_clearsky_ghi(timestamps, lats, lons, solar_zenith_angle=None):
    """
    Compute clear-sky GHI using the correct local time for each site
    or directly from solar zenith angle if available.

    Args:
        timestamps: Either a dictionary of {site_idx -> array of local timestamps}
                   OR a single array of timestamps for all sites
        lats: Array of latitude values
        lons: Array of longitude values
        solar_zenith_angle: Optional pre-computed solar zenith angle values (time, sites)

    Returns:
        clear_sky_ghi: Estimated clear sky GHI values
    """
    # Determine the format of timestamps and handle accordingly
    if isinstance(timestamps, dict):
        # Original format: dictionary of site_idx -> timestamps
        timestamps_dict = timestamps
        n_times = len(next(iter(timestamps_dict.values())))  # Get length from first entry
    else:
        # New format: single array of timestamps for all sites
        n_times = len(timestamps)
        # Create a timestamps dictionary for compatibility with original function
        timestamps_dict = {}
        for site_idx in range(len(lats)):
            timestamps_dict[site_idx] = timestamps

    n_sites = len(lats)
    clear_sky_ghi = np.zeros((n_times, n_sites), dtype=np.float32)
    solar_constant = 1366.1  # W/m²

    # Process based on the type of input available
    if solar_zenith_angle is not None and len(solar_zenith_angle.shape) == 2:
        # First, determine what type of data we're dealing with
        sza_min = np.min(solar_zenith_angle)
        sza_max = np.max(solar_zenith_angle)

        # Process based on data type
        if solar_zenith_angle.dtype in [np.int16, np.uint16, np.int32, np.uint32] and sza_max > 180:
            # Assume it's stored as degrees * 100
            solar_zenith_degrees = solar_zenith_angle.astype(float) / 100.0

            # Calculate cosine of zenith for clear sky GHI calculation
            zenith_rad = np.deg2rad(solar_zenith_degrees)
            cos_zenith = np.cos(zenith_rad)

            # Calculate clear sky GHI for daytime sites (cosine > 0)
            clear_sky_ghi = np.where(cos_zenith > 0,
                                    solar_constant * cos_zenith * 0.7,  # 0.7 is atmospheric transmittance factor
                                    0)
            return clear_sky_ghi
        elif (sza_min >= -1.0 and sza_max <= 1.0):
            # This appears to be cosine of zenith angle directly
            cos_zenith = solar_zenith_angle.astype(float)

            # Calculate clear sky GHI for daytime sites (cosine > 0)
            clear_sky_ghi = np.where(cos_zenith > 0,
                                    solar_constant * cos_zenith * 0.7,
                                    0)
            return clear_sky_ghi
        else:
            # Likely stored as degrees directly
            solar_zenith_degrees = solar_zenith_angle.astype(float)

            # Calculate cosine of zenith
            zenith_rad = np.deg2rad(solar_zenith_degrees)
            cos_zenith = np.cos(zenith_rad)

            # Calculate clear sky GHI for daytime sites
            clear_sky_ghi = np.where(cos_zenith > 0,
                                    solar_constant * cos_zenith * 0.7,
                                    0)
            return clear_sky_ghi

    # Calculate clear sky GHI from scratch using timestamps and coordinates
    print("  Calculating clear sky GHI from timestamps and coordinates")

    # Same calculation as in compute_nighttime_mask but storing clear sky values
    for site_idx in range(n_sites):
        lat = lats[site_idx]
        lon = lons[site_idx]
        local_timestamps = timestamps_dict[site_idx]

        lat_rad = np.deg2rad(lat)

        for t, timestamp in enumerate(local_timestamps):
            # Extract day of year
            doy = timestamp.timetuple().tm_yday

            # Calculate solar declination angle
            decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
            decl_rad = np.deg2rad(decl)

            # Calculate hour angle
            hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
            ha = (hour - 12) * 15  # Hour angle in degrees
            ha_rad = np.deg2rad(ha)

            # Calculate solar zenith angle cosine
            cos_theta = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)

            # For numerical stability, clip to valid range
            cos_theta = max(min(cos_theta, 1.0), -1.0)

            # Calculate clear sky GHI if it's daytime
            if cos_theta > 0:
                clear_sky_ghi[t, site_idx] = solar_constant * cos_theta * 0.7  # 0.7 is atmospheric transmittance factor
            else:
                clear_sky_ghi[t, site_idx] = 0  # Nighttime, no solar radiation

    return clear_sky_ghi


def compute_physical_constraints(timestamps, lats, lons, solar_zenith_angle=None):
    """
    Compute nighttime mask and clear-sky GHI using the correct local time for each site
    or directly from solar zenith angle if available.

    NOTE: This function is kept for backward compatibility.
    Consider using compute_nighttime_mask and compute_clearsky_ghi separately.

    Args:
        timestamps: Either a dictionary of {site_idx -> array of local timestamps}
                   OR a single array of timestamps for all sites
        lats: Array of latitude values
        lons: Array of longitude values
        solar_zenith_angle: Optional pre-computed solar zenith angle values (time, sites)
                           in degrees, where 90° is horizon, <90° is day, >90° is night
                           OR cosine of zenith angle, where 0 is horizon, >0 is day, <0 is night

    Returns:
        nighttime_mask: Boolean mask indicating nighttime (True) or daytime (False)
        clear_sky_ghi: Estimated clear sky GHI values
    """
    # Compute nighttime mask
    nighttime_mask = compute_nighttime_mask(timestamps, lats, lons, solar_zenith_angle)

    # Compute clear sky GHI
    clear_sky_ghi = compute_clearsky_ghi(timestamps, lats, lons, solar_zenith_angle)

    return nighttime_mask, clear_sky_ghi
