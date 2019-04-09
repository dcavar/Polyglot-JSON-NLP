from polyglotjsonnlp import PolyglotPipeline
from pyjsonnlp.microservices.flask_server import FlaskMicroservice

app = FlaskMicroservice(__name__, PolyglotPipeline(), base_route='/')
app.with_constituents = False
app.with_coreferences = False
app.with_dependencies = False
app.with_expressions = True

if __name__ == "__main__":
    app.run(debug=True)