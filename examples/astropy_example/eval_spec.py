# Minimal code path from {'eval.py evaluate_concatenate'} to {'/Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py Angle._wrap_at'}

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/funcs.py:354
def concatenate(coords):
    """
    Combine multiple coordinate objects into a single
    `~astropy.coordinates.SkyCoord`.

    "Coordinate objects" here mean frame objects with data,
    `~astropy.coordinates.SkyCoord`, or representation objects.  Currently,
    they must all be in the same frame, but in a future version this may be
    relaxed to allow inhomogeneous sequences of objects.

    Parameters
    ----------
    coords : sequence of coordinate-like
        The objects to concatenate

    Returns
    -------
    cskycoord : SkyCoord
        A single sky coordinate with its data set to the concatenation of all
        the elements in ``coords``
    """
    if getattr(coords, "isscalar", False) or not isiterable(coords):
        raise TypeError("The argument to concatenate must be iterable")

    scs = [SkyCoord(coord, copy=False) for coord in coords]

    # Check that all frames are equivalent
    for sc in scs[1:]:
        if not sc.is_equivalent_frame(scs[0]):
            raise ValueError(
                f"All inputs must have equivalent frames: {sc} != {scs[0]}"
            )

    # TODO: this can be changed to SkyCoord.from_representation() for a speed
    # boost when we switch to using classmethods
    return SkyCoord(
        concatenate_representations([c.data for c in coords]), frame=scs[0].frame
    )

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/sky_coordinate.py:305 | Class: SkyCoord
def __init__(self, *args, copy=True, **kwargs):
    # these are frame attributes set on this SkyCoord but *not* a part of
    # the frame object this SkyCoord contains
    self._extra_frameattr_names = set()

    # If all that is passed in is a frame instance that already has data,
    # we should bypass all of the parsing and logic below. This is here
    # to make this the fastest way to create a SkyCoord instance. Many of
    # the classmethods implemented for performance enhancements will use
    # this as the initialization path
    if (
        len(args) == 1
        and len(kwargs) == 0
        and isinstance(args[0], (BaseCoordinateFrame, SkyCoord))
    ):
        coords = args[0]
        if isinstance(coords, SkyCoord):
            self._extra_frameattr_names = coords._extra_frameattr_names
            self.info = coords.info

            # Copy over any extra frame attributes
            for attr_name in self._extra_frameattr_names:
                # Setting it will also validate it.
                setattr(self, attr_name, getattr(coords, attr_name))

            coords = coords.frame

        if not coords.has_data:
            raise ValueError(
                "Cannot initialize from a coordinate frame "
                "instance without coordinate data"
            )

        if copy:
            self._sky_coord_frame = coords.copy()
        else:
            self._sky_coord_frame = coords

    else:
        # Get the frame instance without coordinate data but with all frame
        # attributes set - these could either have been passed in with the
        # frame as an instance, or passed in as kwargs here
        frame_cls, frame_kwargs = _get_frame_without_data(args, kwargs)

        # Parse the args and kwargs to assemble a sanitized and validated
        # kwargs dict for initializing attributes for this object and for
        # creating the internal self._sky_coord_frame object
        args = list(args)  # Make it mutable
        skycoord_kwargs, components, info = _parse_coordinate_data(
            frame_cls(**frame_kwargs), args, kwargs
        )

        # In the above two parsing functions, these kwargs were identified
        # as valid frame attributes for *some* frame, but not the frame that
        # this SkyCoord will have. We keep these attributes as special
        # skycoord frame attributes:
        for attr in skycoord_kwargs:
            # Setting it will also validate it.
            setattr(self, attr, skycoord_kwargs[attr])

        if info is not None:
            self.info = info

        # Finally make the internal coordinate object.
        frame_kwargs.update(components)
        self._sky_coord_frame = frame_cls(copy=copy, **frame_kwargs)

        if not self._sky_coord_frame.has_data:
            raise ValueError("Cannot create a SkyCoord without data")

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/sky_coordinate_parsers.py:207
def _parse_coordinate_data(frame, args, kwargs):
    """
    Extract coordinate data from the args and kwargs passed to SkyCoord.

    By this point, we assume that all of the frame attributes have been
    extracted from kwargs (see _get_frame_without_data()), so all that are left
    are (1) extra SkyCoord attributes, and (2) the coordinate data, specified in
    any of the valid ways.
    """
    valid_skycoord_kwargs = {}
    valid_components = {}
    info = None

    # Look through the remaining kwargs to see if any are valid attribute names
    # by asking the frame transform graph:
    attr_names = list(kwargs.keys())
    for attr in attr_names:
        if attr in frame_transform_graph.frame_attributes:
            valid_skycoord_kwargs[attr] = kwargs.pop(attr)

    # By this point in parsing the arguments, anything left in the args and
    # kwargs should be data. Either as individual components, or a list of
    # objects, or a representation, etc.

    # Get units of components
    units = _get_representation_component_units(args, kwargs)

    # Grab any frame-specific attr names like `ra` or `l` or `distance` from
    # kwargs and move them to valid_components.
    valid_components.update(_get_representation_attrs(frame, units, kwargs))

    # Error if anything is still left in kwargs
    if kwargs:
        # The next few lines add a more user-friendly error message to a
        # common and confusing situation when the user specifies, e.g.,
        # `pm_ra` when they really should be passing `pm_ra_cosdec`. The
        # extra error should only turn on when the positional representation
        # is spherical, and when the component 'pm_<lon>' is passed.
        pm_message = ""
        if frame.representation_type == SphericalRepresentation:
            frame_names = list(frame.get_representation_component_names().keys())
            lon_name = frame_names[0]
            lat_name = frame_names[1]

            if f"pm_{lon_name}" in list(kwargs.keys()):
                pm_message = (
                    "\n\n By default, most frame classes expect the longitudinal proper"
                    " motion to include the cos(latitude) term, named"
                    f" `pm_{lon_name}_cos{lat_name}`. Did you mean to pass in this"
                    " component?"
                )

        raise ValueError(
            "Unrecognized keyword argument(s) {}{}".format(
                ", ".join(f"'{key}'" for key in kwargs), pm_message
            )
        )

    # Finally deal with the unnamed args.  This figures out what the arg[0]
    # is and returns a dict with appropriate key/values for initializing
    # frame class. Note that differentials are *never* valid args, only
    # kwargs.  So they are not accounted for here (unless they're in a frame
    # or SkyCoord object)
    if args:
        if len(args) == 1:
            # One arg which must be a coordinate.  In this case coord_kwargs
            # will contain keys like 'ra', 'dec', 'distance' along with any
            # frame attributes like equinox or obstime which were explicitly
            # specified in the coordinate object (i.e. non-default).
            _skycoord_kwargs, _components = _parse_coordinate_arg(
                args[0], frame, units, kwargs
            )

            # Copy other 'info' attr only if it has actually been defined.
            if "info" in getattr(args[0], "__dict__", ()):
                info = args[0].info

        elif len(args) <= 3:
            _skycoord_kwargs = {}
            _components = {}

            frame_attr_names = frame.representation_component_names.keys()
            repr_attr_names = frame.representation_component_names.values()

            for arg, frame_attr_name, repr_attr_name, unit in zip(
                args, frame_attr_names, repr_attr_names, units
            ):
                attr_class = frame.representation_type.attr_classes[repr_attr_name]
                _components[frame_attr_name] = attr_class(arg, unit=unit)

        else:
            raise ValueError(
                f"Must supply no more than three positional arguments, got {len(args)}"
            )

        # The next two loops copy the component and skycoord attribute data into
        # their final, respective "valid_" dictionaries. For each, we check that
        # there are no relevant conflicts with values specified by the user
        # through other means:

        # First validate the component data
        for attr, coord_value in _components.items():
            if attr in valid_components:
                raise ValueError(
                    _conflict_err_msg.format(
                        attr, coord_value, valid_components[attr], "SkyCoord"
                    )
                )
            valid_components[attr] = coord_value

        # Now validate the custom SkyCoord attributes
        for attr, value in _skycoord_kwargs.items():
            if attr in valid_skycoord_kwargs and np.any(
                valid_skycoord_kwargs[attr] != value
            ):
                raise ValueError(
                    _conflict_err_msg.format(
                        attr, value, valid_skycoord_kwargs[attr], "SkyCoord"
                    )
                )
            valid_skycoord_kwargs[attr] = value

    return valid_skycoord_kwargs, valid_components, info

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/sky_coordinate_parsers.py:366
def _parse_coordinate_arg(coords, frame, units, init_kwargs):
    """
    Single unnamed arg supplied.  This must be:
    - Coordinate frame with data
    - Representation
    - SkyCoord
    - List or tuple of:
      - String which splits into two values
      - Iterable with two values
      - SkyCoord, frame, or representation objects.

    Returns a dict mapping coordinate attribute names to values (or lists of
    values)
    """
    from .sky_coordinate import SkyCoord

    is_scalar = False  # Differentiate between scalar and list input
    # valid_kwargs = {}  # Returned dict of lon, lat, and distance (optional)
    components = {}
    skycoord_kwargs = {}

    frame_attr_names = list(frame.representation_component_names.keys())
    repr_attr_names = list(frame.representation_component_names.values())
    repr_attr_classes = list(frame.representation_type.attr_classes.values())
    n_attr_names = len(repr_attr_names)

    # Turn a single string into a list of strings for convenience
    if isinstance(coords, str):
        is_scalar = True
        coords = [coords]

    if isinstance(coords, (SkyCoord, BaseCoordinateFrame)):
        # Note that during parsing of `frame` it is checked that any coordinate
        # args have the same frame as explicitly supplied, so don't worry here.

        if not coords.has_data:
            raise ValueError("Cannot initialize from a frame without coordinate data")

        data = coords.data.represent_as(frame.representation_type)

        # If coords did not have an explicit distance then don't include in initializers.
        if isinstance(coords.data, UnitSphericalRepresentation):
            try:
                index = repr_attr_names.index("distance")
            except ValueError:
                pass
            else:
                del repr_attr_names[index]
                del units[index]
                del frame_attr_names[index]
                del repr_attr_classes[index]

        # List of values corresponding to representation attrs
        values = [getattr(data, name) for name in repr_attr_names]

        if coords.data.differentials and "s" in coords.data.differentials:
            orig_vel = coords.data.differentials["s"]
            vel = coords.data.represent_as(
                frame.representation_type, frame.get_representation_cls("s")
            ).differentials["s"]
            for frname, reprname in frame.get_representation_component_names(
                "s"
            ).items():
                if (
                    reprname == "d_distance"
                    and not hasattr(orig_vel, reprname)
                    and "unit" in orig_vel.get_name()
                ):
                    continue
                values.append(getattr(vel, reprname))
                units.append(None)
                frame_attr_names.append(frname)
                repr_attr_names.append(reprname)
                repr_attr_classes.append(vel.attr_classes[reprname])

        is_skycoord = isinstance(coords, SkyCoord)
        for attr in frame_transform_graph.frame_attributes:
            if (value := getattr(coords, attr, None)) is not None and (
                is_skycoord or attr not in coords.frame_attributes
            ):
                skycoord_kwargs[attr] = value

    elif isinstance(coords, BaseRepresentation):
        if coords.differentials and "s" in coords.differentials:
            diffs = frame.get_representation_cls("s")
            data = coords.represent_as(frame.representation_type, diffs)
            values = [getattr(data, name) for name in repr_attr_names]
            for frname, reprname in frame.get_representation_component_names(
                "s"
            ).items():
                values.append(getattr(data.differentials["s"], reprname))
                units.append(None)
                frame_attr_names.append(frname)
                repr_attr_names.append(reprname)
                repr_attr_classes.append(data.differentials["s"].attr_classes[reprname])

        else:
            data = coords.represent_as(frame.representation_type)
            values = [getattr(data, name) for name in repr_attr_names]

    elif (
        isinstance(coords, np.ndarray)
        and coords.dtype.kind in "if"
        and coords.ndim == 2
        and coords.shape[1] <= 3
    ):
        # 2-d array of coordinate values.  Handle specially for efficiency.
        values = coords.transpose()  # Iterates over repr attrs

    elif isinstance(coords, (Sequence, np.ndarray)):
        # Handles list-like input.
        coord_types = (SkyCoord, BaseCoordinateFrame, BaseRepresentation)
        if any(isinstance(coord, coord_types) for coord in coords):
            # this parsing path is used when there are coordinate-like objects
            # in the list - instead of creating lists of values, we create
            # SkyCoords from the list elements and then combine them.
            scs = [SkyCoord(coord, **init_kwargs) for coord in coords]

            # Check that all frames are equivalent
            for sc in scs[1:]:
                if not sc.is_equivalent_frame(scs[0]):
                    raise ValueError(
                        f"List of inputs don't have equivalent frames: {sc} != {scs[0]}"
                    )

            # Now use the first to determine if they are all UnitSpherical
            not_unit_sphere = not isinstance(scs[0].data, UnitSphericalRepresentation)

            # get the frame attributes from the first coord in the list, because
            # from the above we know it matches all the others.  First copy over
            # the attributes that are in the frame itself, then copy over any
            # extras in the SkyCoord
            for fattrnm in scs[0].frame.frame_attributes:
                skycoord_kwargs[fattrnm] = getattr(scs[0].frame, fattrnm)
            for fattrnm in scs[0]._extra_frameattr_names:
                skycoord_kwargs[fattrnm] = getattr(scs[0], fattrnm)

            # Now combine the values, to be used below
            values = [
                np.concatenate([np.atleast_1d(getattr(sc, data_attr)) for sc in scs])
                for data_attr, repr_attr in zip(frame_attr_names, repr_attr_names)
                if not_unit_sphere or repr_attr != "distance"
            ]
        else:
            is_radec = (
                "ra" in frame.representation_component_names
                and "dec" in frame.representation_component_names
            )
            # none of the elements are "frame-like", create a list of sequences like
            # [[v1_0, v2_0, v3_0], ... [v1_N, v2_N, v3_N]]
            vals = [
                _parse_one_coord_str(c, is_radec=is_radec) if isinstance(c, str) else c
                for c in coords
            ]

            # Do some basic validation of the list elements: all have a length and all
            # lengths the same
            try:
                n_coords = {len(x) for x in vals}
            except Exception as err:
                raise ValueError(
                    "One or more elements of input sequence does not have a length."
                ) from err

            if len(n_coords) > 1:
                raise ValueError(
                    "Input coordinate values must have same number of elements, found"
                    f" {sorted(n_coords)}"
                )
            n_coords = n_coords.pop()

            # Must have no more coord inputs than representation attributes
            if n_coords > n_attr_names:
                raise ValueError(
                    f"Input coordinates have {n_coords} values but representation"
                    f" {frame.representation_type.get_name()} only accepts"
                    f" {n_attr_names}"
                )

            # Now transpose vals to get [(v1_0 .. v1_N), (v2_0 .. v2_N), (v3_0 .. v3_N)]
            # (ok since we know it is exactly rectangular).  (Note: can't just use zip(*values)
            # because Longitude et al distinguishes list from tuple so [a1, a2, ..] is needed
            # while (a1, a2, ..) doesn't work.
            values = [list(x) for x in zip(*vals)]

            if is_scalar:
                values = [x[0] for x in values]
    else:
        raise ValueError("Cannot parse coordinates from first argument")

    # Finally we have a list of values from which to create the keyword args
    # for the frame initialization.  Validate by running through the appropriate
    # class initializer and supply units (which might be None).
    try:
        for frame_attr_name, repr_attr_class, value, unit in zip(
            frame_attr_names, repr_attr_classes, values, units
        ):
            components[frame_attr_name] = repr_attr_class(
                value, unit=unit, copy=COPY_IF_NEEDED
            )
    except Exception as err:
        raise ValueError(
            f'Cannot parse first argument data "{value}" for attribute'
            f" {frame_attr_name}"
        ) from err
    return skycoord_kwargs, components

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py:714 | Class: Longitude
def __new__(cls, angle, unit=None, wrap_angle=None, **kwargs):
    # Forbid creating a Long from a Lat.
    if isinstance(angle, Latitude):
        raise TypeError(
            "A Longitude angle cannot be created from a Latitude angle."
        )
    self = super().__new__(cls, angle, unit=unit, **kwargs)
    if wrap_angle is None:
        wrap_angle = getattr(angle, "wrap_angle", self._default_wrap_angle)
    self.wrap_angle = wrap_angle  # angle-like b/c property setter
    return self

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py:738 | Class: Longitude
def wrap_angle(self):

    return self._wrap_angle

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py:402 | Class: Angle
def _wrap_at(self, wrap_angle):
    """
    Implementation that assumes ``angle`` is already validated
    and that wrapping is inplace.
    """
    # Convert the wrap angle and 360 degrees to the native unit of
    # this Angle, then do all the math on raw Numpy arrays rather
    # than Quantity objects for speed.
    a360 = u.degree.to(self.unit, 360.0)
    wrap_angle = wrap_angle.to_value(self.unit)
    self_angle = self.view(np.ndarray)
    if NUMPY_LT_2_0:
        # Ensure ndim>=1 so that comparison is done using the angle dtype.
        self_angle = self_angle[np.newaxis]
    else:
        # Use explicit float to ensure casting to self_angle.dtype (NEP 50).
        wrap_angle = float(wrap_angle)
    wrap_angle_floor = wrap_angle - a360
    # Do the wrapping, but only if any angles need to be wrapped
    #
    # Catch any invalid warnings from the floor division.
    with np.errstate(invalid="ignore"):
        wraps = (self_angle - wrap_angle_floor) // a360
    valid = np.isfinite(wraps) & (wraps != 0)
    if np.any(valid):
        self_angle -= wraps * a360
        # Rounding errors can cause problems.
        self_angle[self_angle >= wrap_angle] -= a360
        self_angle[self_angle < wrap_angle_floor] += a360