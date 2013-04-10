#!/usr/bin/env python
"""
Module defining the TOD trainer to train the TOD models
"""

from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward
from ecto_opencv import calib, features2d, highgui
from ecto_opencv.features2d import FeatureDescriptor
from object_recognition_core.ecto_cells.db import ModelWriter
from object_recognition_core.pipelines.training import TrainerBase
from object_recognition_tod import ecto_training
import ecto
from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward

########################################################################################################################

class TodTrainer(ecto.BlackBox, TrainerBase):
    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        TrainerBase.__init__(self)

    @classmethod
    def declare_cells(cls, _p):
        # passthrough cells
        cells = {'object_id': CellInfo(ecto.Constant),
                 'json_db': CellInfo(ecto.Constant)}

        # 'real cells'
        cells.update({'model_filler': ecto_training.ModelFiller(),
                      'model_writer': CellInfo(ModelWriter, params={'method':'TOD'}),
                      'trainer': CellInfo(ecto_training.Trainer)})

        return cells

    @classmethod
    def declare_forwards(cls, _p):
        p = {'json_db': [Forward('value', 'json_db')],
             'object_id': [Forward('value', 'object_id')]}
        p.update({'trainer': 'all'})
        i = {}
        o = {}

        return (p,i,o)

    def connections(self, p):
        connections = [ self.object_id[:] >> self.trainer['object_id'],
                        self.json_db[:] >> self.trainer['json_db'] ]
        connections += [ self.trainer['descriptors', 'points'] >> self.model_filler['descriptors', 'points'] ]

        # Connect the model builder to the source
        connections += [ self.object_id[:] >> self.model_writer['object_id'],
                         self.json_db[:] >> self.model_writer['json_db'],
                         self.model_filler['db_document'] >> self.model_writer['db_document']]

        return connections
