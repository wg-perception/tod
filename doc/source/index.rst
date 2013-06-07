:orphan:

.. _tod:

object_recognition_tod: Textured Object Detection
#################################################

Texture Object Detection (TOD) is based on a standard descriptor-matching technique.

Training
********

In the config file you need to specify the feature/descriptor to use as well as the search parameters.

The DB parameters are standard :ref:`ObjectDbParameters <orkcore:object_recognition_core_db>` parameters. A typical config file looks like this:

.. literalinclude:: ../../conf/training.ork
    :language: yaml

During training, in the different views of the object features and descriptors are extracted. For each of those, if depth was also captured (which is the only supported method and is highly recommended anyway), the 3d position is also stored.

You can also view the point cloud of the features by launching the ``apps/feature_viewer`` application

.. program-output:: ../../apps/feature_viewer --help
    :prompt:
    :in_srcdir:

Detection
*********

A typical config file looks like this:

.. literalinclude:: ../../conf/detection.ork
    :language: yaml

During detection, features/descriptors are computed on the current image and compared to our database. Sets of seen descriptors are then checked with the nearest neighbors (descriptor-wise) for an analogous 3d configuration. In the case of 3d input data, it is just a 3d to 3d comparison, but if the input is only 2d, it's a PnP problem (for which we have not plugged the solvePnP from OpenCV yet).

So basically, you can only get the pose of an object on an RGBD input for now.

Limitations
***********

The technique has a few limitations for now:

 * it only supports the features/descriptors defined in OpenCV for now.
 * it does not work with 2d input: it's ``just`` a matter of plugin the solvePnP d'OpenCV but that requires testing
