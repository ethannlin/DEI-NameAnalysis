from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class NameForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=0, max=50)])
    submit = SubmitField('Submit')