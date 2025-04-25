import numpy as np
from datetime import datetime, timedelta


def compute_nighttime_mask(timestamps, lats, lons, solar_zenith_angle=None):
    """
    Compute nighttime mask using the correct local time for each site
    or directly from solar zenith angle if available.

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
    nighttime_mask = np.zeros((n_times, n_sites), dtype=np.float32)

    # If solar zenith angle is already available, use it directly
    if solar_zenith_angle is not None:
        # Make sure we have the right shape
        if len(solar_zenith_angle.shape) == 2:
            # First, determine what type of data we're dealing with
            sza_min = np.min(solar_zenith_angle)
            sza_max = np.max(solar_zenith_angle)
            print(f"  Solar zenith angle data range: {sza_min} to {sza_max} (dtype: {solar_zenith_angle.dtype})")

            # CASE 1: Potentially stored as degrees * 100 (common in int16/uint16 datasets)
            if solar_zenith_angle.dtype in [np.int16, np.uint16, np.int32, np.uint32] and sza_max > 180:
                # Assume it's stored as degrees * 100
                solar_zenith_degrees = solar_zenith_angle.astype(float) / 100.0
                print(f"  Interpreted as degrees * 100, scaled to {np.min(solar_zenith_degrees):.2f}-{np.max(solar_zenith_degrees):.2f}° range")

                # Determine nighttime (solar zenith angle > 90 degrees means sun is below horizon)
                nighttime_mask = (solar_zenith_degrees >= 90.0).astype(np.float32)
                return nighttime_mask

            # CASE 2: Potentially stored as cosine of zenith angle (between -1 and 1)
            elif (sza_min >= -1.0 and sza_max <= 1.0) or (np.abs(sza_min) < 10 and np.abs(sza_max) < 10):
                print(f"  Data appears to be cosine of zenith angle or in unusual units")

                # Carefully handle potential cosine of zenith angle
                if np.all((solar_zenith_angle >= -1.0) & (solar_zenith_angle <= 1.0)):
                    cos_zenith = solar_zenith_angle.astype(float)

                    # If data range is very small around 0-1, it might not be cosine but some other format
                    # In this case, we'll calculate solar position from timestamps as a fallback
                    if sza_max < 0.1 and sza_min > -0.1:
                        print(f"  [WARNING] Solar zenith angle range is very small. Might not be correct format.")
                        print(f"  Falling back to calculating from timestamps...")
                        use_timestamps = True
                    else:
                        use_timestamps = False
                        # For cosine values: cos(zenith) <= 0 means sun is below horizon (nighttime)
                        nighttime_mask = (cos_zenith <= 0).astype(np.float32)
                        print(f"  Using cosine of zenith angle directly, with threshold at 0")
                        return nighttime_mask
                else:
                    print(f"  [WARNING] Values outside valid cosine range [-1, 1]")
                    # Calculate solar position from timestamps as a fallback
                    use_timestamps = True

            # CASE 3: Likely stored as degrees directly
            else:
                solar_zenith_degrees = solar_zenith_angle.astype(float)
                print(f"  Interpreted as degrees directly, range {np.min(solar_zenith_degrees):.2f}-{np.max(solar_zenith_degrees):.2f}°")

                # Determine nighttime (solar zenith angle > 90 degrees means sun is below horizon)
                nighttime_mask = (solar_zenith_degrees >= 90.0).astype(np.float32)
                return nighttime_mask
        else:
            print(f"  [WARNING] solar_zenith_angle has unexpected shape {solar_zenith_angle.shape}, expected 2D array")
            # Continue with calculation from timestamps

    # If we get here, we need to calculate from timestamps and coordinates
    print("  [WARNING] Calculating nighttime mask from timestamps and coordinates")

    # Calculate nighttime mask based on solar position
    for site_idx in range(n_sites):
        lat = lats[site_idx]
        lon = lons[site_idx]
        local_timestamps = timestamps_dict[site_idx]

        lat_rad = np.deg2rad(lat)

        for t, timestamp in enumerate(local_timestamps):
            # Extract day of year
            doy = timestamp.timetuple().tm_yday

            # Calculate solar declination angle
            # Accurate formula for solar declination
            decl = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
            decl_rad = np.deg2rad(decl)

            # Calculate hour angle (angle of sun east or west of local meridian)
            # Use the exact local hour (hour + minutes/60 + seconds/3600)
            hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
            ha = (hour - 12) * 15  # Hour angle in degrees (15 degrees per hour from solar noon)
            ha_rad = np.deg2rad(ha)

            # Calculate solar zenith angle cosine (cosine of angle between sun and zenith)
            cos_theta = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)

            # For numerical stability, clip to valid range [-1, 1]
            cos_theta = max(min(cos_theta, 1.0), -1.0)

            # Determine if it's nighttime (cos_theta <= 0 means sun is below horizon)
            if cos_theta <= 0:
                nighttime_mask[t, site_idx] = 1.0
            else:
                nighttime_mask[t, site_idx] = 0.0

    return nighttime_mask


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
