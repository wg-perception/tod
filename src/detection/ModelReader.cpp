#include <string>

#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <opencv2/core/core.hpp>

#include <ecto/ecto.hpp>

#include <object_recognition_core/common/types.h>
#include <object_recognition_core/db/db.h>
#include <object_recognition_core/db/opencv.h>

using ecto::tendrils;
using object_recognition_core::db::CollectionName;

typedef std::string ModelId;

namespace tod
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** Cell that loads a TOD model from the DB
   */
  struct ModelReader
  {
    static void
    declare_params(tendrils& params)
    {
      params.declare(&ModelReader::db_params_, "db_params", "The DB parameters").required(true);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare < std::string > ("model_id", "The DB id of the model to load.");
      outputs.declare < cv::Mat > ("descriptors", "The descriptors.");
      outputs.declare < std::string > ("object_id", "The DB object ID.");
      outputs.declare < cv::Mat > ("points", "The 3d position of the points.");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      db_params_ = params["db_params"];

      db_ = db_params_->generateDb();
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      const std::string & model_id = inputs.get < std::string > ("model_id");
      object_recognition_core::db::Document doc;
      doc.set_db(db_);
      doc.set_document_id(model_id);
      doc.load_fields();

      cv::Mat points, descriptors;
      doc.get_attachment < cv::Mat > ("points", points);
      doc.get_attachment < cv::Mat > ("descriptors", descriptors);

      outputs["descriptors"] << descriptors;
      outputs["object_id"] << doc.id();
      outputs["points"] << points;

      return ecto::OK;
    }
    object_recognition_core::db::ObjectDbPtr db_;
    ecto::spore<object_recognition_core::db::ObjectDbParameters> db_params_;
  };
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** Cell that loads a TOD model from the DB
   */
  struct ModelReaderIterative
  {
    static void
    declare_params(tendrils& params)
    {
      params.declare<boost::python::object>("model_ids", "The DB id of the model to load.");
      params.declare(&ModelReaderIterative::db_params_, "db_params", "The DB parameters").required(true);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      outputs.declare < std::vector<cv::Mat> > ("points", "The 3d position of the points.");
      outputs.declare < std::vector<cv::Mat> > ("descriptors", "The descriptors.");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      db_params_ = params["db_params"];
      db_ = db_params_->generateDb();

      const boost::python::object & python_object_ids = params.get < boost::python::object > ("object_ids");
      boost::python::stl_input_iterator<std::string> begin(python_object_ids), end;
      std::copy(begin, end, std::back_inserter(model_ids_));
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      // Load the list of models to load

      std::vector<cv::Mat> point_vector, descriptor_vector;
      std::vector < std::string > object_ids;

      BOOST_FOREACH(const ModelId & model_id, model_ids_)
      {
        object_recognition_core::db::Document doc;
        doc.set_db(db_);
        doc.set_document_id(model_id);
        doc.load_fields();

        cv::Mat descriptors, points;
        doc.get_attachment<cv::Mat>("descriptors", descriptors);
        doc.get_attachment<cv::Mat>("points", points);

        descriptor_vector.push_back(descriptors);
        object_ids.push_back(doc.id());
        point_vector.push_back(points);
      }

      outputs["descriptors"] << descriptor_vector;
      outputs["object_ids"] << object_ids;
      outputs["points"] << point_vector;

      return ecto::OK;
    }
    object_recognition_core::db::ObjectDbPtr db_;
    ecto::spore<object_recognition_core::db::ObjectDbParameters> db_params_;
    std::vector<ModelId> model_ids_;
  };
}

ECTO_CELL(ecto_detection, tod::ModelReader, "ModelReader", "Reads a TOD model from the db")
ECTO_CELL(ecto_detection, tod::ModelReaderIterative, "ModelReaderIterative",
    "Reads several TOD models from the db")
