import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask import session, send_file
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from werkzeug.utils import secure_filename
import os
from os.path import join, dirname, basename, splitext
import pandas as pd
from config import config
from models.ModelUser import ModelUser
from forms import UserForm, UpdateUserForm, ChangePasswordForm
from models.entities.User import User, Rol
from werkzeug.security import generate_password_hash
from models.entities.Dataset import Dataset
from models.entities.Preprocesamiento import Preprocesamiento
from models.entities.Preppred import Preppred 
from models.entities.Trainxgboost import Trainxgboost 
from models.entities.Predxgboost import Predxgboost
from models.entities.Trainrnn import Trainrnn
from models.entities.Trainhybrid import Trainhybrid
from sqlalchemy.orm import joinedload
from utils.preprocessing import realizar_preprocesamiento
from utils.preprocessingpred import realizar_preprocesamiento_pred
from utils.trainxgboost import realizar_xgboost
from utils.guardarxgboost import guardar_modelo
from utils.guardarrnn import guardar_modelo_rnn
from utils.cargar_modelo import cargar_modelo
from utils.cargar_modelo_rnn import cargar_modelo_rnn
from utils.cargar_modelo_hyb import cargar_modelo_hxgboost, cargar_modelo_hrnn, predecir_con_modelo_hibrido
from utils.trainrnn import realizar_rnn
from utils.trainhybrid import realizar_hyb
from utils.guardarhyb import guardar_modelo_hibrido
from extensions import db
from dash import Dash, html, dcc, no_update
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from functools import wraps
from sqlalchemy import inspect, text

def requires_roles(*roles):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Verifica si el usuario está autenticado y si su rol está en la lista de roles permitidos
            if not current_user.is_authenticated or current_user.rol.Nombre not in roles:
                flash("No tienes permiso para acceder a esta página.", "warning")
                return redirect(url_for('home'))
            print(f"Usuario actual: {current_user.User}, Rol: {current_user.rol.Nombre}")
            if current_user.rol.Nombre not in roles:
                flash("No tienes permiso para acceder a esta página.", "warning")
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        return wrapped
    return wrapper

app = Flask(__name__)
app.config.from_object(config['production'])

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def protect_dash_views(dash_app):
    for view_func in dash_app.server.view_functions:
        if view_func.startswith(dash_app.config.routes_pathname_prefix):
            dash_app.server.view_functions[view_func] = login_required(dash_app.server.view_functions[view_func])

def roles_required(view, roles):
    @wraps(view)
    def decorated_view(*args, **kwargs):
        if not current_user.is_authenticated:
            print("Redirigiendo al login, usuario no autenticado.")
            return redirect(url_for('login'))
        if current_user.rol.Nombre not in roles:
            print(f"Acceso denegado, rol del usuario: {current_user.rol.Nombre}")
            flash("No tienes permiso para acceder a esta página.", "warning")
            return redirect(url_for('home'))
        return view(*args, **kwargs)
    return decorated_view

# Crear una instancia de Dash integrada con tu aplicación Flask
dash_app = Dash(__name__, server=app, external_stylesheets=['/static/css/stylesd.css'], url_base_pathname='/visual/')

# Proteger todas las rutas de Dash
protect_dash_views(dash_app)

@login_manager.user_loader
def load_user(user_id):
    session['update_predictions'] = True
    print("Loaded User:", User.query.get(int(user_id)))
    return User.query.get(int(user_id))

app.secret_key = '895399e34bd2a307747f137851c529b8'
app.config['UPLOAD_FOLDER'] = 'src/dataset'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
db.init_app(app)

with app.app_context():
    @dash_app.callback(
        Output('pred-dropdown', 'options'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_dropdown_options(n):
        # Verificar si hay una actualización pendiente y si es así, actualizar la lista
        if session.get('update_predictions', False):
            all_predictions = Predxgboost.query.with_entities(Predxgboost.NombrePrediccion).all()
            print(all_predictions)
            session['update_predictions'] = False  # Resetear la señal
            return [{'label': prediction.NombrePrediccion, 'value': prediction.NombrePrediccion} for prediction in all_predictions]
        return no_update

    dash_app.layout = html.Div([
        dcc.Interval(id='interval-component', interval=60000, n_intervals=0),
        html.A("Volver a Home", href="/home", className="btn btn-login-login mb-3"),
        html.H1("Dashboard de Visualización"),
        dcc.Dropdown(
            id='pred-dropdown',
            options=[],
            placeholder="Seleccione una Predicción"
        ),
        dcc.Dropdown(
            id='column-dropdown',
            options=[],
            multi=True,
            placeholder="Seleccione Columnas para Gráficos"
        ),
        html.Div(id='dashboard-content'),
        html.Div(id='graph-container'),
        dcc.Dropdown(
            id='groupby-dropdown',
            options=[],
            placeholder="Seleccione Columna para Agrupar en Gráfico de Barras"
        ),
        html.Div(id='groupby-graph-container'),
        dcc.Dropdown(
            id='piechart-dropdown',
            options=[],
            placeholder="Seleccione Columna para Gráfico Circular"
        ),
        html.Div(id='piechart-container'),
        dcc.Input(
            id='meta-input',
            type='number',
            min=1,
            max=100,
            step=1,
            placeholder='Ingrese Meta de Churn (%)'
        ),
        html.Div(id='gauge-chart-container')
    ])

# Callback de Dash para actualizar el dashboard
@dash_app.callback(
    Output('dashboard-content', 'children'),
    [Input('pred-dropdown', 'value')]
)

def update_dashboard(selected_prediction):
    if not selected_prediction:
        return 'Por favor, selecciona una predicción.'

    prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
    if not prediction_record:
        return 'No se encontró el archivo de predicción seleccionado.'

    # Obtener la ruta relativa del archivo de predicción
    pred_file_path = prediction_record.PredFilePath

    # Agregar el prefijo 'src/' a la ruta relativa
    abs_file_path = os.path.join('src', pred_file_path)

    # Leer los datos del archivo CSV
    try:
        df = pd.read_csv(abs_file_path)
    except FileNotFoundError:
        return 'El archivo de predicción no pudo ser encontrado.'

    # Calcula los porcentajes de churn y permanencia
    churn_rate = (df.iloc[:, -1] == 1).mean() * 100
    stay_rate = 100 - churn_rate

    return html.Div([
        html.H3(f'Porcentaje de Churn: {churn_rate:.2f}%'),
        html.H3(f'Porcentaje de Permanencia: {stay_rate:.2f}%')
    ])

# Callback para actualizar las opciones de los dropdowns
@dash_app.callback(
    [Output('column-dropdown', 'options'),
     Output('column-dropdown', 'value'),
     Output('groupby-dropdown', 'options'),
     Output('groupby-dropdown', 'value'),
     Output('piechart-dropdown', 'options'),
     Output('piechart-dropdown', 'value')],
    [Input('pred-dropdown', 'value')]
)

def update_dropdowns(selected_prediction):
    if selected_prediction:
        prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
        if prediction_record:
            abs_file_path = os.path.join('src', prediction_record.PredFilePath)
            try:
                df = pd.read_csv(abs_file_path)
                columns_options = [{'label': col, 'value': col} for col in df.columns]
                return columns_options, None, columns_options, None, columns_options, None
            except FileNotFoundError:
                pass
    return [], None, [], None, [], None

# Callback para crear gráficos basados en columnas seleccionadas
@dash_app.callback(
    Output('graph-container', 'children'),
    [Input('column-dropdown', 'value'),
     Input('pred-dropdown', 'value')]
)
def update_graph(selected_columns, selected_prediction):
    if not selected_prediction or not selected_columns:
        return 'Seleccione una predicción y al menos una columna para visualizar.'

    prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
    pred_file_path = os.path.join('src', prediction_record.PredFilePath)
    df = pd.read_csv(pred_file_path)

    # Gráfico de barras para una columna
    if len(selected_columns) == 1:
        fig = px.bar(df, x=selected_columns[0], title=f'Distribución de {selected_columns[0]}')
        return dcc.Graph(figure=fig)

    # Swarm plot para dos columnas
    elif len(selected_columns) == 2:
        fig = px.strip(df, x=selected_columns[0], y=selected_columns[1], title=f'Relación entre {selected_columns[0]} y {selected_columns[1]}')
        return dcc.Graph(figure=fig)

    # Gráfico 3D para tres columnas
    elif len(selected_columns) == 3:
        fig = px.scatter_3d(df, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2], title=f'Relación entre {selected_columns[0]}, {selected_columns[1]} y {selected_columns[2]}')
        return dcc.Graph(figure=fig)

    return 'Seleccione un máximo de tres columnas para visualizar.'

# Callback para el gráfico de barras agrupado
@dash_app.callback(
    Output('groupby-graph-container', 'children'),
    [Input('groupby-dropdown', 'value'),
     Input('pred-dropdown', 'value')]
)
def update_groupby_graph(groupby_column, selected_prediction):
    if not selected_prediction or not groupby_column:
        return 'Seleccione una predicción y una columna para agrupar.'

    # Obtener la ruta relativa del archivo de predicción y leer los datos
    prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
    pred_file_path = os.path.join('src', prediction_record.PredFilePath)
    df = pd.read_csv(pred_file_path)

    # Crear el gráfico de barras agrupado
    grouped_data = df.groupby(groupby_column).size().reset_index(name='counts')
    fig = px.bar(grouped_data, x=groupby_column, y='counts', title=f'Distribución de {groupby_column}')
    return dcc.Graph(figure=fig)

# Callback para el gráfico circular
@dash_app.callback(
    Output('piechart-container', 'children'),
    [Input('piechart-dropdown', 'value'),
     Input('pred-dropdown', 'value')]
)
def update_piechart(piechart_column, selected_prediction):
    if not selected_prediction or not piechart_column:
        return 'Seleccione una predicción y una columna para el gráfico circular.'

    # Obtener la ruta relativa del archivo de predicción y leer los datos
    prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
    pred_file_path = os.path.join('src', prediction_record.PredFilePath)
    df = pd.read_csv(pred_file_path)

    # Crear el DataFrame para el gráfico circular
    pie_data = df[piechart_column].value_counts().reset_index()
    pie_data.columns = ['category', 'count']

    # Imprimir el DataFrame para diagnosticar el problema
    print(pie_data)

    # Crear el gráfico circular
    fig = px.pie(pie_data, values='count', names='category', title=f'Distribución de {piechart_column}')
    return dcc.Graph(figure=fig)

# Callback para el gauge chart
@dash_app.callback(
    Output('gauge-chart-container', 'children'),
    [Input('meta-input', 'value'),
     Input('pred-dropdown', 'value')]
)
def update_gauge_chart(meta_input, selected_prediction):
    if not selected_prediction or meta_input is None:
        return 'Ingrese una meta y seleccione una predicción.'

    prediction_record = Predxgboost.query.filter_by(NombrePrediccion=selected_prediction).first()
    pred_file_path = os.path.join('src', prediction_record.PredFilePath)
    df = pd.read_csv(pred_file_path)

    # Calcular porcentaje de churn
    churn_rate = (df.iloc[:, -1] == 1).mean() * 100

    # Determinar el color del gauge en función de qué tan cerca está del objetivo
    if churn_rate > meta_input + 10:
        gauge_color = 'red'  # Lejos de la meta, excede por 10 o más
    elif churn_rate > meta_input:
        gauge_color = 'yellow'  # Moderadamente cerca de la meta, excede por menos de 10
    else:
        gauge_color = 'green'  # Cerca o igual a la meta

    # Crear el gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        title={'text': "Churn Rate vs Meta"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': gauge_color}}
    ))
    return dcc.Graph(figure=fig)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash("Por favor, rellena todos los campos.")
            return render_template('auth/login.html')

        logged_user = ModelUser.authenticate(username, password)

        if logged_user:
            login_user(logged_user)  # Log the user in
            return redirect(url_for('home'))
        else:
            flash("Credenciales Inválidas")
            return render_template('auth/login.html')
    else:
        return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Esto limpia la sesión
    flash('Has cerrado sesión correctamente.', 'success')
    return redirect(url_for('login'))

@app.route('/download_file/')
def download_file():
    try:
        path = request.args.get('path')
        source = request.args.get('source')
        # Preponer 'src/' a la ruta obtenida
        full_path = os.path.join('src', path) if path else None
        if full_path:
            return send_file(full_path, as_attachment=True)
        elif source == 'reporte':
            flash("Seleccione una predicción para descargar", 'error')
            return redirect(url_for('reporte'))
        else:
            return "File path not provided", 400
    except Exception as e:
        flash(f"Error al descargar el archivo: {str(e)}", 'error')
        return redirect(url_for('home'))
    

@login_manager.unauthorized_handler
def unauthorized():
    flash('Debes iniciar sesión para acceder a esta página.', 'warning')
    return redirect(url_for('login'))

def create_default_roles():
    with app.app_context():
        existing_roles = Rol.query.count()
        if existing_roles == 0:
            roles = ['Administrador', 'Analista', 'Gerente']
            for role_name in roles:
                new_role = Rol(Nombre=role_name)
                db.session.add(new_role)
            db.session.commit()
            print("Default roles created.")

def create_default_admin():
    with app.app_context():
        if User.query.count() == 0:
            admin_role = Rol.query.filter_by(Nombre='Administrador').first()
            if not admin_role:
                # Crear rol de administrador si no existe
                admin_role = Rol(Nombre='Administrador')
                db.session.add(admin_role)
                db.session.commit()

            # Asignar el hash del password proporcionado directamente
            password_hash = "1234"
            admin_user = User(
                username='Admin',  # Cambiado de User a username para coincidir con el constructor de la clase User
                password=password_hash,  # Se pasa el hash directamente sin rehashing
                email='master@upc.edu.pe',
                rol=admin_role
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin created.")

@app.route('/home')
@login_required  # Protect this route with login_required
def home():
    return render_template('home.html')

def delete_user(form=None):
    user_id = request.form.get('user_id_to_delete')
    if user_id:
        user_to_delete = User.query.get(user_id)
        if user_to_delete:
            db.session.delete(user_to_delete)
            db.session.commit()
            flash('Usuario eliminado exitosamente', 'success')
        else:
            flash('Error al eliminar el usuario', 'error')
    else:
        flash('Selecciona un usuario para eliminar', 'warning')

def update_user(form):
    user_id = request.form.get('user_id_to_update')
    if user_id:
        user_to_update = User.query.get(int(user_id))
        if user_to_update:
            user_to_update.User = request.form['username']
            user_to_update.Email = request.form['email']
            selected_rol = Rol.query.get(request.form['rol'])
            user_to_update.rol = selected_rol
            db.session.commit()
            flash('Usuario actualizado exitosamente', 'success')
            return user_id
        else:
            flash('Error al actualizar el usuario', 'error')
    else:
        flash('Selecciona un usuario para actualizar', 'warning')

def change_password(form):
    form = ChangePasswordForm(request.form)
    user_id = request.form.get('user_id')
    if user_id:
        user_to_change_password = User.query.get(int(user_id))
        if user_to_change_password and user_to_change_password.check_password(request.form['current_password']):
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            
            if new_password == confirm_password:
                user_to_change_password.Password = generate_password_hash(new_password, method='pbkdf2:sha256')
                db.session.commit()
                flash('Contraseña actualizada exitosamente', 'success')
                return user_id
            else:
                flash('La nueva contraseña y la confirmación no coinciden', 'error')
        else:
            flash('Error al cambiar la contraseña', 'error')
    else:
        flash('Selecciona un usuario para cambiar la contraseña', 'warning')

def add_user(form):
    existing_user_by_username = User.query.filter_by(User=form.User.data).first()
    existing_user_by_email = User.query.filter_by(Email=form.Email.data).first()

    if existing_user_by_username:
        flash('Ese nombre de usuario ya existe. Por favor, elige otro.', 'error')
    elif existing_user_by_email:
        flash('Ese correo electrónico ya está registrado. Por favor, utiliza otro.', 'error')
    else:
        selected_rol = Rol.query.get(form.rol.data)
        nuevo_usuario = User(
            username=form.User.data,
            password=form.Password.data,
            email=form.Email.data,
            rol=selected_rol
        )
        db.session.add(nuevo_usuario)
        db.session.commit()
        flash('Usuario añadido exitosamente', 'success')

# Diccionario de acciones
ACTIONS = {
    'delete': delete_user,
    'update': update_user,
    'add': add_user,
    'change_password': change_password
}

@app.route('/gestion_usuarios', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador')
def gestion_usuarios():
    try:
        print("Request Method:", request.method)
        print("POST Data:", request.form)
        if request.method == 'POST':
            action = request.form.get('action', 'add')
        else:
            action = request.args.get('action', 'add')

        print("Action:", action)    
        roles = Rol.query.all()
        usuarios = User.query.all()
        user_to_update = None

        form = UserForm()
        form.rol.choices = [(rol.ID, rol.Nombre) for rol in Rol.query.all()]

        if action == 'change_password':
            form = ChangePasswordForm()
            user_to_update = None 
            user_id = request.form.get('selected_user_id') or request.args.get('user_id')
            if user_id:
                user_to_update = User.query.get(user_id)
                form = ChangePasswordForm(obj=user_to_update)
                return render_template('gestion_usuarios/actpssw.html', form=form, users=usuarios, user_to_update=user_to_update)

        elif action == 'update':
            user_id = request.form.get('selected_user_id') or request.args.get('user_id')
            print("Captured user_id:", user_id)
            if user_id:
                print("User ID recibido:", user_id)
                user_to_update = User.query.get(user_id)
                print("Usuario seleccionado para actualizar:", user_to_update)
                print("User ID:", user_id)
                print("User to Update:", user_to_update)
            form = UpdateUserForm(obj=user_to_update)
            form.rol.choices = [(rol.ID, rol.Nombre) for rol in Rol.query.all()]
        else:
            form = UserForm()
            form.rol.choices = [(rol.ID, rol.Nombre) for rol in Rol.query.all()]

        # Añadir lógica para manejar la selección del usuario
        if request.method == 'POST' and 'select_user' in request.form:
            user_id = request.form['user_id']
            return redirect(url_for('gestion_usuarios', action='update', user_id=user_id))

        # Si es POST, ejecuta la función correspondiente
        if request.method == 'POST':
            action_function = ACTIONS.get(action)
            if action_function:
                result = action_function(form)  # <-- Obtener el resultado de la función
                if action == 'update' and result:  # <-- Comprobar si el resultado existe (es decir, si hay un user_id)
                    return redirect(url_for('gestion_usuarios', action='update', user_id=result))

        usuarios = User.query.all()

        if action == 'delete':
            return render_template('gestion_usuarios/elimuser.html', users=usuarios)
        elif action == 'update':
            return render_template('gestion_usuarios/actuser.html', form=form, users=usuarios, roles=roles, user_to_update=user_to_update)
        elif action == 'change_password':
            return render_template('gestion_usuarios/actpssw.html', form=form, users=usuarios, user_to_update=user_to_update)    
        else:
            return render_template('gestion_usuarios/useradm.html', form=form, users=usuarios)
        
    except Exception as e:
        flash(f"Error durante la gestión de usuarios: {str(e)}", 'error')
        return redirect(url_for('home'))

@app.route('/carga_datos', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def carga_datos():
    if not current_user.is_authenticated:
        flash('Necesitas Loguearte Primero')
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Verifica si el post request tiene el archivo parte
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Si el usuario no selecciona archivo, el navegador también
        # envía un archivo vacío sin nombre.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Extracción del nombre del dataset del formulario
        dataset_name = request.form['dataset_name'].strip()
        
        # Verificar que el nombre del dataset es único
        existing_dataset = Dataset.query.filter_by(Nombre=dataset_name).first()
        if existing_dataset:
            flash('Un dataset con ese nombre ya existe. Por favor, elige otro nombre.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Renombrar el archivo con el nombre del dataset
            filename = secure_filename(dataset_name) + '.' + file.filename.rsplit('.', 1)[1].lower()
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.User)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            
            dataset_description = request.form['dataset_description']

            new_dataset = Dataset(
                Nombre=dataset_name,
                Descripción=dataset_description,
                UsuarioID=current_user.id,
                Fecha=datetime.datetime.now(),
                FilePath=file_path
            )
            db.session.add(new_dataset)
            db.session.commit()

            flash('Archivo subido con éxito')
            return redirect(url_for('carga_datos'))
        else:
            flash('Solo se permiten archivos csv')
            return redirect(request.url)
        
    datasets = Dataset.query.options(joinedload(Dataset.usuario)).all()
    return render_template('carga_datos/subirdb.html', datasets=datasets)

@app.route('/carga_datos/eliminar', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def carga_datos_eliminar():
    datasets = Dataset.query.all()
    if request.method == 'POST':
        dataset_id = request.form.get('selected_dataset')
        if dataset_id:
            dataset = Dataset.query.get(dataset_id)
            if dataset:
                try:
                    # Elimina el archivo del sistema de archivos
                    if os.path.exists(dataset.FilePath):
                        os.remove(dataset.FilePath)
                    # Elimina la referencia del dataset en la base de datos
                    db.session.delete(dataset)
                    db.session.commit()
                    flash('Dataset eliminado correctamente', 'success')
                except Exception as e:
                    db.session.rollback()
                    flash('Hubo un error al eliminar el dataset', 'error')
            else:
                flash('Dataset no encontrado', 'error')
        else:
            flash('Por favor, selecciona un dataset para eliminar', 'warning')
    
    datasets = Dataset.query.all()
    return render_template('carga_datos/elimdb.html', datasets=datasets)

@app.route('/preprocesamiento', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def preprocesamiento():
    datasets = Dataset.query.all()  # Cargar todos los datasets
    preview_data = pd.DataFrame()   # Inicializar preview_data como un DataFrame vacío
    selected_features = []          # Inicializar selected_features como lista vacía
    # Obtener la lista de preprocesamientos realizados
    preprocesamientos = Preprocesamiento.query \
                        .join(User, Preprocesamiento.UsuarioID == User.ID) \
                        .join(Dataset, Preprocesamiento.DatasetID == Dataset.ID) \
                        .add_columns(User.User, Dataset.Nombre, Preprocesamiento.Fecha, Preprocesamiento.Comentario) \
                        .all()

    if request.method == 'POST':
        dataset_id = request.form.get('selected_dataset')
        if not dataset_id:  # Comprobar si se seleccionó un dataset
            flash('Por favor seleccione un dataset', 'error')  # Mostrar mensaje de error
            return render_template('modelado_predictivo/preproc.html', datasets=datasets, preview_data=preview_data, preprocesamientos=preprocesamientos, selected_features=selected_features)
        
        try:
            dataset = Dataset.query.get(dataset_id)

            if dataset:
                user_folder = os.path.join('src/preproc', current_user.User)
                if not os.path.exists(user_folder):
                    os.makedirs(user_folder)

                # Crear una subcarpeta con la fecha actual
                date_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                base_folder = os.path.join(user_folder, date_folder)
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)

                # Ejecutar el preprocesamiento
                preprocessed_data, selected_features, num_rows, num_features, x_train_file_path, x_test_file_path, y_train_file_path, y_test_file_path = realizar_preprocesamiento(dataset.FilePath, base_folder)

                preprocessed_file_path = os.path.join(base_folder, dataset.Nombre + '_preproc.csv')
                preprocessed_data.to_csv(preprocessed_file_path, index=False)

                # Guardar en base de datos
                new_preproc = Preprocesamiento(
                    DatasetID=dataset.ID,
                    UsuarioID=current_user.ID,
                    Fecha=datetime.datetime.now(),
                    Comentario="Preprocesamiento completado",
                    FilePath=preprocessed_file_path,
                    X_trainPath=x_train_file_path,
                    X_testPath=x_test_file_path,
                    y_trainPath=y_train_file_path,
                    y_testPath=y_test_file_path
                )
                db.session.add(new_preproc)
                db.session.commit()
                
                # Cargar las primeras 10 filas para la vista previa
                preview_data = pd.read_csv(x_train_file_path).head(10)

                # Recargar la lista de preprocesamientos para asegurar que incluye el último realizado
                preprocesamientos = Preprocesamiento.query \
                                .join(User, Preprocesamiento.UsuarioID == User.ID) \
                                .join(Dataset, Preprocesamiento.DatasetID == Dataset.ID) \
                                .add_columns(User.User, Dataset.Nombre, Preprocesamiento.Fecha, Preprocesamiento.Comentario) \
                                .all()
                flash('Preprocesamiento realizado con éxito.', 'success')
            else:
                flash('No se encontró el dataset seleccionado.', 'error')
                return redirect(url_for('home'))
        except Exception as e:
            flash(f'Error al preprocesar el dataset: {str(e)}', 'error')
            return redirect(url_for('home'))

    return render_template('modelado_predictivo/preproc.html', datasets=datasets, preview_data=preview_data, preprocesamientos=preprocesamientos, selected_features=selected_features)

@app.route('/backup', methods=['GET'])
@login_required
@requires_roles('Administrador')
def backup():
    try:
        def get_db_tables():
            inspector = inspect(db.engine)
            table_names = inspector.get_table_names()
            table_data = {}
            for table_name in table_names:
                query = text(f"SELECT * FROM {table_name}")
                rows = db.session.execute(query).fetchall()
                table_data[table_name] = [row._asdict() for row in rows]
            return table_data

        table_data = get_db_tables()
        return render_template('gestion_usuarios/backup.html', table_data=table_data)
    except Exception as e:
        print("Error durante la recuperación de las tablas:", str(e))
        return str(e), 500

@app.route('/download_db_backup')
def download_db_backup():
    db_path = 'retainai.db'  # Verifica que esta ruta es correcta y ajusta si es necesario
    directory = os.path.join(os.getcwd(), 'instance')  # Ajusta si tu archivo está en 'instance' o en otro directorio
    return send_from_directory(directory, db_path, as_attachment=True, download_name='backup_retainai.db')

@app.route('/preprocesamiento_prediccion', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def preprocesamiento_prediccion():

    try:

        user_folder = os.path.join('src/dataset_pred', current_user.User)
        date_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_folder = os.path.join(user_folder, date_folder)
        preview_data = pd.DataFrame()
        
        
        if request.method == 'POST':
            file = request.files.get('file')
            selected_preproc_id = request.form.get('selected_dataset')

            # Verificar si no se seleccionó un dataset preprocesado
            if not selected_preproc_id:
                flash('Por favor seleccione un dataset', 'error')
                return redirect(url_for('preprocesamiento_prediccion'))

            if file and allowed_file(file.filename):
                if not os.path.exists(base_folder):
                        os.makedirs(base_folder)

                filename = secure_filename(file.filename)
                file_path = os.path.join(base_folder, filename)
                file.save(file_path)
                session['uploaded_file_path'] = file_path
                flash('Archivo subido correctamente', 'success')
            else:
                flash('Archivo no permitido', 'error')
            # Verificar si se seleccionó un dataset preprocesado
            if selected_preproc_id:
                selected_preproc = Preprocesamiento.query.get(selected_preproc_id)
                if selected_preproc and selected_preproc.X_trainPath:
                    session['selected_x_train_path'] = selected_preproc.X_trainPath

            # Ejecutar preprocesamiento personalizado
            if file and selected_preproc_id:
                uploaded_file_path = session.get('uploaded_file_path')
                selected_x_train_path = session.get('selected_x_train_path')

                if uploaded_file_path and selected_x_train_path:
                    df_pred, num_rows, num_columns = realizar_preprocesamiento_pred(uploaded_file_path, selected_x_train_path, base_folder)
                    dataset_name_with_pred = filename.rsplit('.', 1)[0] + '_pred'

                    if df_pred is not None:
                        # Construir el mensaje de éxito
                        preprocessing_message = f"Preprocesamiento realizado con éxito. Filas: {num_rows}, Columnas: {num_columns}"

                        # Guardar el DataFrame preprocesado en un archivo CSV
                        preprocessed_file_path = os.path.join(base_folder, dataset_name_with_pred + '.csv')
                        df_pred.to_csv(preprocessed_file_path, index=False)

                        fecha = datetime.datetime.strptime(date_folder, '%Y-%m-%d_%H-%M-%S')
                        
                        # Crear un nuevo registro en la tabla Preppred
                        new_preppred = Preppred(
                            UsuarioID=current_user.ID,
                            Fecha = fecha,
                            Comentario=f"Preprocesado para predicción completado. Filas: {num_rows}, Columnas: {num_columns}",
                            FilePath=preprocessed_file_path,
                            Nombre=dataset_name_with_pred
                        )
                        db.session.add(new_preppred)
                        db.session.commit()

                        flash(preprocessing_message, 'success')
                    else:
                        flash(preprocessing_message, 'error')

        preprocesamientos = Preprocesamiento.query \
                            .join(User, Preprocesamiento.UsuarioID == User.ID) \
                            .join(Dataset, Preprocesamiento.DatasetID == Dataset.ID) \
                            .add_columns(Preprocesamiento.ID, User.User, Dataset.Nombre, Preprocesamiento.Fecha, Preprocesamiento.Comentario) \
                            .all()
        
        preppreds = Preppred.query \
                        .join(User, Preppred.UsuarioID == User.ID) \
                        .add_columns(Preppred.Nombre, User.User, Preppred.Fecha, Preppred.Comentario) \
                        .all()

        return render_template('modelado_predictivo/preprocpred.html', preprocesamientos=preprocesamientos, datasets_listos=preppreds,preview_data=preview_data)
    
    except Exception as e:
        flash(str(e), 'error')
        return redirect(url_for('home'))

@app.route('/trainxgboost', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def train_xgboost():
    try:    
        preprocesamientos = Preprocesamiento.query \
                            .options(joinedload(Preprocesamiento.dataset), joinedload(Preprocesamiento.usuario)) \
                            .all()
        entrenamientos_xgboost = Trainxgboost.query \
                                .join(User, Trainxgboost.UsuarioID == User.ID) \
                                .add_columns(User.User, Trainxgboost.ModeloNombre, Trainxgboost.Fecha, Trainxgboost.Accuracy, Trainxgboost.Recall, Trainxgboost.F1Score) \
                                .all()
        resultados = None
        
        if request.method == 'POST':
            preproc_id = request.form.get('selected_preproc')
            if not preproc_id:
                flash('Por favor selecciona un dataset para entrenamiento.', 'error')
                return redirect(url_for('train_xgboost'))  # redirige de nuevo a la misma página para que el usuario pueda seleccionar
            
            preprocesamiento = Preprocesamiento.query.get(preproc_id)

            if preprocesamiento:
                # Obtener las rutas de los archivos
                x_train_path = preprocesamiento.X_trainPath
                x_test_path = preprocesamiento.X_testPath
                y_train_path = preprocesamiento.y_trainPath
                y_test_path = preprocesamiento.y_testPath

                # Ejecutar el entrenamiento de XGBoost
                modelo, accuracy, recall, f1_score, conf_matrix, report = realizar_xgboost(x_train_path, x_test_path, y_train_path, y_test_path)

                # Guardar resultados para mostrar en la vista
                resultados = {
                    'conf_matrix': conf_matrix,
                    'report': report,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1_score': f1_score
                }

                # Obtener el nombre del dataset
                dataset_nombre = preprocesamiento.dataset.Nombre

                # Obtener la fecha actual
                fecha_actual = datetime.datetime.now()

                # Guardar el modelo y los resultados
                model_path = guardar_modelo(modelo, current_user.User, dataset_nombre, accuracy)

                # Extraer el nombre del archivo del modelo
                model_name = os.path.basename(model_path)

                # (Necesitas definir y crear el modelo Trainxgboost en tu módulo models)
                new_train = Trainxgboost(
                    UsuarioID=current_user.ID,
                    DatasetID=preprocesamiento.DatasetID,
                    Accuracy=accuracy,
                    Recall=recall,
                    F1Score=f1_score,
                    ModeloPath=model_path,
                    ModeloNombre=model_name,
                    Fecha=fecha_actual
                )
                db.session.add(new_train)
                db.session.commit()

                flash('Entrenamiento realizado con éxito', 'success')
                return redirect(url_for('train_xgboost'))
            
        # Al recargar la página, los datos actualizados se mostrarán aquí
        entrenamientos_xgboost_actualizados = Trainxgboost.query \
                                .join(User, Trainxgboost.UsuarioID == User.ID) \
                                .add_columns(User.User, Trainxgboost.ModeloNombre, Trainxgboost.Fecha, Trainxgboost.Accuracy, Trainxgboost.Recall, Trainxgboost.F1Score) \
                                .all()

        return render_template('modelado_predictivo/trainxgboost.html', preprocesamientos=preprocesamientos, entrenamientos_xgboost=entrenamientos_xgboost_actualizados, resultados=resultados)
    except Exception as e:
        flash(f"Error durante el entrenamiento: {str(e)}", 'error')
        return redirect(url_for('home'))

@app.route('/pred_xgboost', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def pred_xgboost():
    try:
        entrenamientos_xgboost = Trainxgboost.query.all()
        preppreds = Preppred.query.all()
        predicciones = Predxgboost.query.all()
        download_path = session.get('download_path', None)  # Obtener la ruta de descarga desde la sesión

        if request.method == 'GET':
            session.pop('download_path', None)  # Remueve la ruta de descarga de la sesión si existe

        if request.method == 'POST':
            selected_model_id = request.form.get('selected_model')
            selected_preppred_id = request.form.get('selected_preppred')
            session['update_predictions'] = True

            if not selected_model_id or not selected_preppred_id:
                flash('Debe seleccionar un modelo y un dataset para la predicción.', 'error')
                return redirect(url_for('pred_xgboost'))

            modelo = Trainxgboost.query.get(selected_model_id)
            preppred = Preppred.query.get(selected_preppred_id)

            if modelo and preppred:
                # Cargar el modelo
                loaded_model = cargar_modelo(modelo.ModeloPath)

                # Cargar el dataset para realizar la predicción
                pred_dataset = pd.read_csv(preppred.FilePath)

                # Realizar predicción
                churn_predictions = loaded_model.predict(pred_dataset)

                # Cargar el dataset original (sin el sufijo _pred) para agregar la predicción
                file_dir = dirname(preppred.FilePath)
                file_name, file_ext = splitext(basename(preppred.FilePath))
                original_file_name = file_name.replace('_pred', '')
                original_file_path = join(file_dir, original_file_name + file_ext)

                original_dataset = pd.read_csv(original_file_path)

                # Asegurarse de que ambos datasets tienen el mismo número de filas
                if len(churn_predictions) != len(original_dataset):
                    flash('Error: El número de predicciones no coincide con el número de filas en el dataset original.', 'error')
                    return redirect(url_for('pred_xgboost'))

                # Agregar la columna de predicción al dataset original
                original_dataset['churn'] = churn_predictions

                #Fecha actual
                fecha_actual = datetime.datetime.now()

                # Carpeta de usuario y fecha
                user_folder = os.path.join('src/pred_xgboost', current_user.User)
                user_folder1 = os.path.join('pred_xgboost', current_user.User)
                date_folder = fecha_actual.strftime('%Y-%m-%d_%H-%M-%S')
                base_folder = os.path.join(user_folder, date_folder)

                base_folderb = os.path.join(user_folder1, date_folder)

                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)

                # Guardar el dataset con predicciones en la ubicación original
                pred_dataset.to_csv(original_file_path, index=False)

                # Guardar el nuevo dataset
                new_filename = f"{modelo.ModeloNombre}_{preppred.Nombre.replace('_pred', '')}.csv"
                new_file_path = os.path.join(base_folder, new_filename)
                new_file_pathb = os.path.join(base_folderb, new_filename)
                original_dataset.to_csv(new_file_path, index=False)

                # Registrar la predicción en la base de datos
                nueva_prediccion = Predxgboost(
                    UsuarioID=current_user.ID,
                    ModeloID=modelo.ID,
                    FilePath=new_file_path,
                    Fecha = fecha_actual,
                    Accuracy=modelo.Accuracy,
                    NombrePrediccion=new_filename,
                    PredFilePath=new_file_pathb
                )
                db.session.add(nueva_prediccion)
                db.session.commit()

                download_path = new_file_pathb
                session['download_path'] = download_path
                flash('Predicción realizada con éxito.', 'success')

            # Asegúrate de recargar la lista de predicciones después de añadir la nueva
            predicciones = Predxgboost.query.all()

        else:
            predicciones = Predxgboost.query.all()

        # Pasar download_path desde la sesión si existe
        download_path = session.get('download_path', None)

        return render_template('modelado_predictivo/predxgboost.html', 
                            entrenamientos_xgboost=entrenamientos_xgboost, 
                            preppreds=preppreds, 
                            predicciones=predicciones,
                            download_path=download_path)
    except Exception as e:
        flash(f"Error durante la predicción con XGBoost: {str(e)}", 'error')
        return redirect(url_for('home'))


@app.route('/trainrnn', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def train_rnn():
    try:    

        preprocesamientos = Preprocesamiento.query \
                            .options(joinedload(Preprocesamiento.dataset), joinedload(Preprocesamiento.usuario)) \
                            .all()
        
        if request.method == 'POST':
            preproc_id = request.form.get('selected_preproc')

            if not preproc_id:
                    flash('Por favor selecciona un dataset para entrenamiento.', 'error')
                    return redirect(url_for('train_rnn'))

            preprocesamiento = Preprocesamiento.query.get(preproc_id)

            if preprocesamiento:
                # Obtener las rutas de los archivos
                x_train_path = preprocesamiento.X_trainPath
                x_test_path = preprocesamiento.X_testPath
                y_train_path = preprocesamiento.y_trainPath
                y_test_path = preprocesamiento.y_testPath

                # Ejecutar el entrenamiento de RNN
                modelo, accuracy, recall, f1_score, conf_matrix, report = realizar_rnn(x_train_path, x_test_path, y_train_path, y_test_path)

                # Guardar resultados para mostrar en la vista
                resultados = {
                    'conf_matrix': conf_matrix,
                    'report': report,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1_score': f1_score
                }

                # Obtener el nombre del dataset
                dataset_nombre = preprocesamiento.dataset.Nombre

                # Obtener la fecha actual
                fecha_actual = datetime.datetime.now()

                # Guardar el modelo y los resultados
                model_path = guardar_modelo_rnn(modelo, current_user.User, dataset_nombre, accuracy)

                # Extraer el nombre del archivo del modelo
                model_name = os.path.basename(model_path)

                new_train = Trainrnn(
                    UsuarioID=current_user.ID,
                    DatasetID=preprocesamiento.DatasetID,
                    Accuracy=accuracy,
                    Recall=recall,
                    F1Score=f1_score,
                    ModeloPath=model_path,
                    ModeloNombre=model_name,
                    Fecha=fecha_actual
                )
                db.session.add(new_train)
                db.session.commit()

                flash('Entrenamiento realizado con éxito', 'success')
                return redirect(url_for('train_rnn'))
            
        # Al recargar la página, los datos actualizados se mostrarán aquí
        entrenamientos_rnn_actualizados = Trainrnn.query \
                                .join(User, Trainrnn.UsuarioID == User.ID) \
                                .add_columns(User.User, Trainrnn.ModeloNombre, Trainrnn.Fecha, Trainrnn.Accuracy, Trainrnn.Recall, Trainrnn.F1Score) \
                                .all()

        return render_template('modelado_predictivo/trainrnn.html', preprocesamientos=preprocesamientos, entrenamientos_rnn=entrenamientos_rnn_actualizados)
    
    except Exception as e:
        flash(f"Error durante el entrenamiento: {str(e)}", 'error')
        return redirect(url_for('home'))
    
@app.route('/pred_rnn', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def pred_rnn():
    try:
        entrenamientos_xgboost = Trainrnn.query.all()
        preppreds = Preppred.query.all()
        predicciones = Predxgboost.query.all()
        download_path = session.get('download_path', None)  # Obtener la ruta de descarga desde la sesión

        if request.method == 'GET':
            session.pop('download_path', None)  # Remueve la ruta de descarga de la sesión si existe

        if request.method == 'POST':
            selected_model_id = request.form.get('selected_model')
            selected_preppred_id = request.form.get('selected_preppred')
            session['update_predictions'] = True

            if not selected_model_id or not selected_preppred_id:
                flash('Debe seleccionar un modelo y un dataset para la predicción.', 'error')
                return redirect(url_for('pred_xgboost'))

            modelo = Trainrnn.query.get(selected_model_id)
            preppred = Preppred.query.get(selected_preppred_id)

            if modelo and preppred:
                # Cargar el modelo
                loaded_model = cargar_modelo_rnn(modelo.ModeloPath)

                # Cargar el dataset para realizar la predicción
                pred_dataset = pd.read_csv(preppred.FilePath)
                pred_dataset_rnn = np.reshape(pred_dataset.values, (pred_dataset.shape[0], 1, pred_dataset.shape[1]))

                # Realizar predicción
                churn_predictions_probabilities = loaded_model.predict(pred_dataset_rnn)

                # Aplicar umbral para convertir probabilidades en clasificaciones binarias
                churn_classifications = (churn_predictions_probabilities > 0.5).astype(int)

                # Cargar el dataset original (sin el sufijo _pred) para agregar la predicción
                file_dir = dirname(preppred.FilePath)
                file_name, file_ext = splitext(basename(preppred.FilePath))
                original_file_name = file_name.replace('_pred', '')
                original_file_path = join(file_dir, original_file_name + file_ext)

                original_dataset = pd.read_csv(original_file_path)

                # Asegurarse de que ambos datasets tienen el mismo número de filas
                if len(churn_classifications) != len(original_dataset):
                    flash('Error: El número de predicciones no coincide con el número de filas en el dataset original.', 'error')
                    return redirect(url_for('pred_rnn'))

                # Agregar la columna de predicción al dataset original
                original_dataset['churn'] = churn_classifications.squeeze()

                #Fecha actual
                fecha_actual = datetime.datetime.now()

                # Carpeta de usuario y fecha
                user_folder = os.path.join('src/pred_xgboost', current_user.User)
                user_folder1 = os.path.join('pred_xgboost', current_user.User)
                date_folder = fecha_actual.strftime('%Y-%m-%d_%H-%M-%S')
                base_folder = os.path.join(user_folder, date_folder)

                base_folderb = os.path.join(user_folder1, date_folder)

                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)

                # Guardar el dataset con predicciones en la ubicación original
                pred_dataset.to_csv(original_file_path, index=False)

                # Guardar el nuevo dataset
                new_filename = f"{modelo.ModeloNombre}_{preppred.Nombre.replace('_pred', '')}.csv"
                new_file_path = os.path.join(base_folder, new_filename)
                new_file_pathb = os.path.join(base_folderb, new_filename)
                original_dataset.to_csv(new_file_path, index=False)

                # Registrar la predicción en la base de datos
                nueva_prediccion = Predxgboost(
                    UsuarioID=current_user.ID,
                    ModeloID=modelo.ID,
                    FilePath=new_file_path,
                    Fecha = fecha_actual,
                    Accuracy=modelo.Accuracy,
                    NombrePrediccion=new_filename,
                    PredFilePath=new_file_pathb
                )
                db.session.add(nueva_prediccion)
                db.session.commit()

                download_path = new_file_pathb
                session['download_path'] = download_path
                flash('Predicción realizada con éxito.', 'success')

            # Asegúrate de recargar la lista de predicciones después de añadir la nueva
            predicciones = Predxgboost.query.all()

        else:
            predicciones = Predxgboost.query.all()

    # Pasar download_path desde la sesión si existe
        download_path = session.get('download_path', None)

        return render_template('modelado_predictivo/predrnn.html', 
                            entrenamientos_xgboost=entrenamientos_xgboost, 
                            preppreds=preppreds, 
                            predicciones=predicciones,
                            download_path=download_path)
    except Exception as e:
        flash(f"Error durante la predicción con RNN: {str(e)}", 'error')
        return redirect(url_for('home'))

@app.route('/train_hyb', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def train_hyb():
    try:    
    
        preprocesamientos = Preprocesamiento.query \
                            .options(joinedload(Preprocesamiento.dataset), joinedload(Preprocesamiento.usuario)) \
                            .all()
        
        if request.method == 'POST':
            preproc_id = request.form.get('selected_preproc')
            if not preproc_id:
                    flash('Por favor selecciona un dataset para entrenamiento.', 'error')
                    return redirect(url_for('train_hyb'))
            
            preprocesamiento = Preprocesamiento.query.get(preproc_id)

            if preprocesamiento:
                # Obtener las rutas de los archivos
                x_train_path = preprocesamiento.X_trainPath
                x_test_path = preprocesamiento.X_testPath
                y_train_path = preprocesamiento.y_trainPath
                y_test_path = preprocesamiento.y_testPath

                # Ejecutar el entrenamiento de RNN
                xgb_model, rnn_model, accuracy, recall, f1_score, conf_matrix, report = realizar_hyb(x_train_path, x_test_path, y_train_path, y_test_path)

                # Guardar resultados para mostrar en la vista
                resultados = {
                    'conf_matrix': conf_matrix,
                    'report': report,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f1_score': f1_score
                }

                # Obtener el nombre del dataset
                dataset_nombre = preprocesamiento.dataset.Nombre

                # Obtener la fecha actual
                fecha_actual = datetime.datetime.now()

                # Guardar modelo xgb y los resultados
                model_path_xgb, model_path_rnn = guardar_modelo_hibrido(xgb_model, rnn_model, current_user.User, dataset_nombre, accuracy)

                # Extraer el nombre del archivo del modelo para cada modelo
                model_name_xgb = os.path.basename(model_path_xgb)
                model_name_rnn = os.path.basename(model_path_rnn)

                model_name_hybrid = f"{model_name_xgb}_AND_{model_name_rnn}"

                new_train = Trainhybrid(
                    UsuarioID=current_user.ID,
                    DatasetID=preprocesamiento.DatasetID,
                    Accuracy=accuracy,
                    Recall=recall,
                    F1Score=f1_score,
                    ModeloxgbPath=model_path_xgb,
                    ModelornnPath=model_path_rnn,
                    ModeloNombre=model_name_hybrid,
                    Fecha=fecha_actual
                )
                db.session.add(new_train)
                db.session.commit()

                flash('Entrenamiento realizado con éxito', 'success')
                return redirect(url_for('train_hyb'))
            
        # Al recargar la página, los datos actualizados se mostrarán aquí
        entrenamientos_hyb_actualizados = Trainhybrid.query \
                                .join(User, Trainhybrid.UsuarioID == User.ID) \
                                .add_columns(User.User, Trainhybrid.ModeloNombre, Trainhybrid.Fecha, Trainhybrid.Accuracy, Trainhybrid.Recall, Trainhybrid.F1Score) \
                                .all()

        return render_template('modelado_predictivo/trainhybrid.html', preprocesamientos=preprocesamientos, entrenamientos_hyb=entrenamientos_hyb_actualizados)
    
    except Exception as e:
        flash(f"Error durante el entrenamiento: {str(e)}", 'error')
        return redirect(url_for('home'))

@app.route('/pred_hyb', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista')
def pred_hyb():
    try:
        entrenamientos_hybrid = Trainhybrid.query.all()
        preppreds = Preppred.query.all()
        predicciones = Predxgboost.query.all()
        download_path = session.get('download_path', None)  # Obtener la ruta de descarga desde la sesión

        if request.method == 'GET':
            session.pop('download_path', None)  # Remueve la ruta de descarga de la sesión si existe
        
        if request.method == 'POST':
            selected_model_id = request.form.get('selected_model')
            selected_preppred_id = request.form.get('selected_preppred')
            session['update_predictions'] = True

            if not selected_model_id or not selected_preppred_id:
                flash('Debe seleccionar un modelo y un dataset para la predicción.', 'error')
                return redirect(url_for('pred_hyb'))

            modelo = Trainhybrid.query.get(selected_model_id)
            preppred = Preppred.query.get(selected_preppred_id)

            if modelo and preppred:
                # Cargar el modelo
                xgb_model_path = modelo.ModeloxgbPath
                rnn_model_path = modelo.ModelornnPath
                pred_dataset = pd.read_csv(preppred.FilePath)
                resultados_prediccion = predecir_con_modelo_hibrido(xgb_model_path, rnn_model_path, pd.read_csv(preppred.FilePath))

                # Cargar el dataset original (sin el sufijo _pred) para agregar la predicción
                file_dir = dirname(preppred.FilePath)
                file_name, file_ext = splitext(basename(preppred.FilePath))
                original_file_name = file_name.replace('_pred', '')
                original_file_path = join(file_dir, original_file_name + file_ext)

                original_dataset = pd.read_csv(original_file_path)

                # Asegurarse de que ambos datasets tienen el mismo número de filas
                if len(resultados_prediccion) != len(original_dataset):
                    flash('Error: El número de predicciones no coincide con el número de filas en el dataset original.', 'error')
                    return redirect(url_for('pred_hyb'))

                # Agregar la columna de predicción al dataset original
                original_dataset['churn'] = resultados_prediccion

                #Fecha actual
                fecha_actual = datetime.datetime.now()

                # Carpeta de usuario y fecha
                user_folder = os.path.join('src/pred_xgboost', current_user.User)
                user_folder1 = os.path.join('pred_xgboost', current_user.User)
                date_folder = fecha_actual.strftime('%Y-%m-%d_%H-%M-%S')
                base_folder = os.path.join(user_folder, date_folder)

                base_folderb = os.path.join(user_folder1, date_folder)

                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)

                # Guardar el dataset con predicciones en la ubicación original
                pred_dataset.to_csv(original_file_path, index=False)

                # Guardar el nuevo dataset
                new_filename = f"{modelo.ModeloNombre}_{preppred.Nombre.replace('_pred', '')}.csv"
                new_file_path = os.path.join(base_folder, new_filename)
                new_file_pathb = os.path.join(base_folderb, new_filename)
                original_dataset.to_csv(new_file_path, index=False)

                # Registrar la predicción en la base de datos
                nueva_prediccion = Predxgboost(
                    UsuarioID=current_user.ID,
                    ModeloID=modelo.ID,
                    FilePath=new_file_path,
                    Fecha = fecha_actual,
                    Accuracy=modelo.Accuracy,
                    NombrePrediccion=new_filename,
                    PredFilePath=new_file_pathb
                )
                db.session.add(nueva_prediccion)
                db.session.commit()

                download_path = new_file_pathb
                session['download_path'] = download_path
                flash('Predicción realizada con éxito.', 'success')

            # Asegúrate de recargar la lista de predicciones después de añadir la nueva
            predicciones = Predxgboost.query.all()

        else:
            predicciones = Predxgboost.query.all()

        # Pasar download_path desde la sesión si existe
        download_path = session.get('download_path', None)

        return render_template('modelado_predictivo/predhybrid.html', 
                            entrenamientos_hybrid=entrenamientos_hybrid, 
                            preppreds=preppreds, 
                            predicciones=predicciones,
                            download_path=download_path)
    except Exception as e:
        flash(f"Error durante la predicción híbrida: {str(e)}", 'error')
        return redirect(url_for('home'))

@app.route('/reporte', methods=['GET', 'POST'])
@login_required
@requires_roles('Administrador', 'Analista', 'Gerente')
def reporte():
    try:
        predicciones = Predxgboost.query.all()
        if request.method == 'POST':
            selected_path = request.form.get('path')
            if not selected_path:
                flash("Seleccione una predicción para descargar", 'error')
                return render_template('visualizacion/reporte.html', predicciones=predicciones)
            return redirect(url_for('download_file', path=selected_path, source='reporte'))
        return render_template('visualizacion/reporte.html', predicciones=predicciones)
    except Exception as e:
        flash(f"Error al cargar el reporte: {str(e)}", 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    create_default_roles()  # Asegura que existan los roles por defecto
    create_default_admin()  # Asegura que exista un administrador por defecto
    app.run()
