from app import app
from flask import redirect, render_template, url_for
from app.models import NameForm
from app.torch_utils import get_prediction

@app.route('/', methods=['GET', 'POST'])
@app.route('/<name>/<result>', methods=['GET', 'POST'])
def index(name=None,result=None):
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data.title()
        return redirect(url_for('predict', name=name), code=307)
    if name and result:
        return render_template('index.html', form=form, name=name, result=result)
    return render_template('index.html', form=form)

@app.route('/predict/<name>', methods=['POST'])
def predict(name):
    output = get_prediction(name)
    # return jsonify({'name': name, 'result': output})
    return redirect(url_for('index', name=name, result=output))