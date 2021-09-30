find_path(Iridescence_INCLUDE_DIRS glk/drawable.hpp
  HINTS /usr/local/include /usr/include
  DOC "Iridescence include directories")

find_library(Iridescence_LIBRARIES NAMES iridescence
  HINTS /usr/local/lib /usr/lib
  DOC "Iridescence libraries")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Iridescence DEFAULT_MSG Iridescence_INCLUDE_DIRS Iridescence_LIBRARIES)
