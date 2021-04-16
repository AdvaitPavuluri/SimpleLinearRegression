from flask import Flask, jsonify, request, Response
from flask_restplus import Resource, Api
from simple_linear_regression import SimpleLR
from werkzeug.datastructures import FileStorage
# initialize
app= Flask(__name__)                #  Create a Flask WSGI application
api = Api(app)                      #  Create a Flask-RESTPlus API

name_space = api.namespace('HousePrices', description='Simple Linear Regression')

upload_parser = api.parser()
upload_parser.add_argument('Train File',
                           location='files',
                           type=FileStorage)
upload_parser.add_argument('Test File', required=False,default=None,
                           location='files',
                           type=FileStorage)


@name_space.route("/predicthouseprice", methods=["POST"])
@name_space.expect(upload_parser)
class MapDataDomain(Resource):
    def post(self):
        if request.method == 'POST':
            args = upload_parser.parse_args()
            train_csv = args.get('Train File')
            test_csv = args.get('Test File')
            slr = SimpleLR(train_csv, test_csv)
            model = slr.train()
            results = slr.test(model)
            return jsonify(results.tolist())


if __name__ == '__main__':
   app.run(port=4455, debug=True)
