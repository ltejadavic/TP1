from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, PasswordField, HiddenField
from wtforms.validators import DataRequired, Email

class UserForm(FlaskForm):
    User = StringField('User', validators=[DataRequired()])
    Password = PasswordField('Contraseña', validators=[DataRequired()])
    Email = StringField('Email', validators=[DataRequired(), Email()])
    rol = SelectField('Rol', coerce=int)  # Las opciones se llenarán dinámicamente
    submit = SubmitField('Añadir Usuario')
    action = HiddenField()
    
class UpdateUserForm(FlaskForm):
    username = StringField('User', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    rol = SelectField('Rol', coerce=int)
    submit = SubmitField('Actualizar')

class ChangePasswordForm(FlaskForm):
    user_id = HiddenField('User ID')
    current_password = PasswordField('Contraseña Actual', validators=[DataRequired()])
    new_password = PasswordField('Nueva Contraseña', validators=[DataRequired()])
    confirm_password = PasswordField('Confirmar Nueva Contraseña', validators=[DataRequired()])
    submit = SubmitField('Cambiar Contraseña')