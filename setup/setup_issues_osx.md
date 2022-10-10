# Mac Setup Issues
There may be issues in installing SUMO dependencies because SUMO version 1.1.0 is a bit old and Brew may have updated versions of dependencies since then. Here we provide solutions to version issues due to several libraries.

In general, if you encounter dependency issue for some library, try looking for a corresponding `<name>.rb` file in `setup/` and run `brew install setup/<name>.rb`.

## libgdal
You may get the error
```
dyld: Symbol not found: __Z33OGRCreateCoordinateTransformationP19OGRSpatialReferenceS0_
  Referenced from: /Users/<user>/sumo_binaries/bin/netconvert
  Expected in: /usr/local/opt/gdal/lib/libgdal.20.dylib
```
Try `brew install setup/gdal.rb` to install the correct version.

## libfox
If you encounter issues with visualization in SUMO GUI (`sumo-gui` command), try reverting to an older version of libfox: `brew install setup/gdal.rb`