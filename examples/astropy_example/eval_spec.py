# Minimal code path from {'evaluate_concatenate'} to {'_wrap_at'}

# Location: eval.py:18
@funsearch.run
def evaluate_concatenate():
    start_time = time.time()
    icrs_array = ICRS(
            ra=np.random.random(10000) * u.deg, dec=np.random.random(10000) * u.deg
    )
    concatenate((icrs_array, icrs_array))
    end_time = time.time()
    return end_time - start_time

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py:398 | Class: Angle
@funsearch.evolve
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

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/angles/core.py:732 | Class: Longitude
def wrap_angle(self, value):
        self._wrap_angle = Angle(value, copy=False)
        self._wrap_at(self.wrap_angle)

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/units/quantity.py:430 | Class: Quantity
def __new__(
        cls: type[Self],
        value: QuantityLike,
        unit=None,
        dtype=np.inexact,
        copy=True,
        order=None,
        subok=False,
        ndmin=0,
    ) -> Self:
        if unit is not None:
            # convert unit first, to avoid multiple string->unit conversions
            unit = Unit(unit)

        # inexact -> upcast to float dtype
        float_default = dtype is np.inexact
        if float_default:
            dtype = None

        copy = sanitize_copy_arg(copy)

        # optimize speed for Quantity with no dtype given, copy=COPY_IF_NEEDED
        if isinstance(value, Quantity):
            if unit is not None and unit is not value.unit:
                value = value.to(unit)
                # the above already makes a copy (with float dtype)
                copy = COPY_IF_NEEDED

            if type(value) is not cls and not (subok and isinstance(value, cls)):
                value = value.view(cls)

            if float_default and value.dtype.kind in "iu":
                dtype = float

            return np.array(
                value, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin
            )

        # Maybe str, or list/tuple of Quantity? If so, this may set value_unit.
        # To ensure array remains fast, we short-circuit it.
        value_unit = None
        if not isinstance(value, np.ndarray):
            if isinstance(value, str):
                # The first part of the regex string matches any integer/float;
                # the second parts adds possible trailing .+-, which will break
                # the float function below and ensure things like 1.2.3deg
                # will not work.
                pattern = (
                    r"\s*[+-]?"
                    r"((\d+\.?\d*)|(\.\d+)|([nN][aA][nN])|"
                    r"([iI][nN][fF]([iI][nN][iI][tT][yY]){0,1}))"
                    r"([eE][+-]?\d+)?"
                    r"[.+-]?"
                )

                v = re.match(pattern, value)
                unit_string = None
                try:
                    value = float(v.group())

                except Exception:
                    raise TypeError(
                        f'Cannot parse "{value}" as a {cls.__name__}. It does not '
                        "start with a number."
                    )

                unit_string = v.string[v.end() :].strip()
                if unit_string:
                    value_unit = Unit(unit_string)
                    if unit is None:
                        unit = value_unit  # signal no conversion needed below.

            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if all(isinstance(v, Quantity) for v in value):
                    # If a list/tuple contains only quantities, convert all
                    # to the same unit.
                    if unit is None:
                        unit = value[0].unit
                    value = [q.to_value(unit) for q in value]
                    value_unit = unit  # signal below that conversion has been done
                elif (
                    dtype is None
                    and not hasattr(value, "dtype")
                    and isinstance(unit, StructuredUnit)
                ):
                    # Special case for list/tuple of values and a structured unit:
                    # ``np.array(value, dtype=None)`` would treat tuples as lower
                    # levels of the array, rather than as elements of a structured
                    # array, so we use the structure of the unit to help infer the
                    # structured dtype of the value.
                    dtype = unit._recursively_get_dtype(value)

        using_default_unit = False
        if value_unit is None:
            # If the value has a `unit` attribute and if not None
            # (for Columns with uninitialized unit), treat it like a quantity.
            value_unit = getattr(value, "unit", None)
            if value_unit is None:
                # Default to dimensionless for no (initialized) unit attribute.
                if unit is None:
                    using_default_unit = True
                    unit = cls._default_unit
                value_unit = unit  # signal below that no conversion is needed
            else:
                try:
                    value_unit = Unit(value_unit)
                except Exception as exc:
                    raise TypeError(
                        f"The unit attribute {value.unit!r} of the input could "
                        "not be parsed as an astropy Unit."
                    ) from exc

                if unit is None:
                    unit = value_unit
                elif unit is not value_unit:
                    copy = COPY_IF_NEEDED  # copy will be made in conversion at end

        value = np.array(
            value, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin
        )

        # For no-user-input unit, make sure the constructed unit matches the
        # structure of the data.
        if using_default_unit and value.dtype.names is not None:
            unit = value_unit = _structured_unit_like_dtype(value_unit, value.dtype)

        # check that array contains numbers or long int objects
        if value.dtype.kind in "OSU" and not (
            value.dtype.kind == "O" and isinstance(value.item(0), numbers.Number)
        ):
            raise TypeError("The value must be a valid Python or Numpy numeric type.")

        # by default, cast any integer, boolean, etc., to float
        if float_default and value.dtype.kind in "iuO":
            value = value.astype(float)

        # if we allow subclasses, allow a class from the unit.
        if subok:
            qcls = getattr(unit, "_quantity_class", cls)
            if issubclass(qcls, cls):
                cls = qcls

        value = value.view(cls)
        value._set_unit(value_unit)
        if unit is value_unit:
            return value
        else:
            # here we had non-Quantity input that had a "unit" attribute
            # with a unit different from the desired one.  So, convert.
            return value.to(unit)

# Location: /Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy/astropy/coordinates/sky_coordinate.py:303 | Class: SkyCoord
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