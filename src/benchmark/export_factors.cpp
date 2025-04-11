#include <gtsam/base/serialization.h>
#include <gtsam/linear/GaussianFactorGraph.h>

// Export GTSAM classes for serialization
BOOST_CLASS_EXPORT_GUID(gtsam::GaussianFactorGraph, "gtsam::GaussianFactorGraph")
BOOST_CLASS_EXPORT_GUID(gtsam::JacobianFactor, "gtsam::JacobianFactor");
BOOST_CLASS_EXPORT_GUID(gtsam::HessianFactor, "gtsam::HessianFactor");