from flask import Flask, json, g, request, jsonify, json
import morphological
app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def morphologically_analyze():
    json_data = json.loads(request.data)
    morphological_instance = morphological.Morphological()
    response=morphological_instance.morphologic_analyze(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response



if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)