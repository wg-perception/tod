#!/usr/bin/env python
"""
Module defining the TOD detector to find objects in a scene
"""

from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward
from ecto_image_pipeline.base import RescaledRegisteredDepth
from ecto_opencv import features2d, highgui, imgproc, calib
from ecto_opencv.calib import DepthTo3d
from ecto_opencv.features2d import FeatureDescriptor
from object_recognition_core.pipelines.detection import DetectorBase
from object_recognition_tod import ecto_detection
import ecto

class TodDetector(ecto.BlackBox, DetectorBase):
    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @staticmethod
    def declare_cells(p):
        guess_params = {}
        guess_params['visualize'] = p.visualize
        guess_params['db'] = p.json_db

        cells = {'depth_map': RescaledRegisteredDepth(),
                 'feature_descriptor': CellInfo(FeatureDescriptor),
                 'guess_generator': CellInfo(ecto_detection.GuessGenerator, guess_params),
                 'passthrough': CellInfo(ecto.PassthroughN, {'items':{'image':'An image', 'K_image':'The camera matrix'}})}

        return cells

    @classmethod
    def declare_forwards(cls, _p):
        p = {'feature_descriptor': [Forward('json_feature_params'),
                                    Forward('json_descriptor_params')],
             'guess_generator': [Forward('n_ransac_iterations'),
                                 Forward('min_inliers'),
                                 Forward('sensor_error')]}
        i = {'passthrough': [Forward('image'), Forward('K_image')],
             'feature_descriptor': [Forward('mask')],
             'depth_map': [Forward('depth')]}

        o = {'feature_descriptor': [Forward('keypoints')],
             'guess_generator': [Forward('pose_results')]}

        return (p, i, o)

    @classmethod
    def declare_direct_params(self, p):
        p.declare('json_db', 'The DB to get data from as a JSON string', '{}')
        p.declare('search', 'The search parameters as a JSON string', '{}')
        p.declare('json_object_ids', 'The ids of the objects to find as a JSON list or the keyword "all".', 'all')
        p.declare('visualize', 'If true, some windows pop up to see the progress', False)

    def configure(self, p, _i, _o):
        self.descriptor_matcher = ecto_detection.DescriptorMatcher("Matcher",
                            search_json_params=p['search'],
                            json_db=p['json_db'],
                            json_object_ids=p['json_object_ids'])

        self._points3d = DepthTo3d()

    def connections(self, p):
        # Rescale the depth image and convert to 3d
        graph = [ self.passthrough['image'] >> self.depth_map['image'],
                  self.depth_map['depth'] >> self._points3d['depth'],
                  self.passthrough['K_image'] >> self._points3d['K'],
                  self._points3d['points3d'] >> self.guess_generator['points3d'] ]
        # make sure the inputs reach the right cells
        if 'depth' in self.feature_descriptor.inputs.keys():
            graph += [ self.depth_map['depth'] >> self.feature_descriptor['depth']]

        graph += [ self.passthrough['image'] >> self.feature_descriptor['image'],
                   self.passthrough['image'] >> self.guess_generator['image'] ]

        graph += [ self.descriptor_matcher['spans'] >> self.guess_generator['spans'],
                   self.descriptor_matcher['object_ids'] >> self.guess_generator['object_ids'] ]

        graph += [ self.feature_descriptor['keypoints'] >> self.guess_generator['keypoints'],
                   self.feature_descriptor['descriptors'] >> self.descriptor_matcher['descriptors'],
                   self.descriptor_matcher['matches', 'matches_3d'] >> self.guess_generator['matches', 'matches_3d'] ]

        cvt_color = imgproc.cvtColor(flag=imgproc.RGB2GRAY)

        if p.visualize:
            draw_keypoints = features2d.DrawKeypoints()
            graph += [ self.passthrough['image'] >> cvt_color[:],
                           cvt_color[:] >> draw_keypoints['image'],
                           self.feature_descriptor['keypoints'] >> draw_keypoints['keypoints']
                           ]

        if p.visualize:
            # visualize the found keypoints
            image_view = highgui.imshow(name="RGB")
            keypoints_view = highgui.imshow(name="Keypoints")

            graph += [ self.passthrough['image'] >> image_view['image'],
                       draw_keypoints['image'] >> keypoints_view['image']
                           ]

            pose_view = highgui.imshow(name="Pose")
            pose_drawer = calib.PosesDrawer()

            # draw the poses
            graph += [ self.passthrough['image', 'K'] >> pose_drawer['image', 'K'],
                       self.guess_generator['Rs', 'Ts'] >> pose_drawer['Rs', 'Ts'],
                       pose_drawer['output'] >> pose_view['image'] ]

        return graph
