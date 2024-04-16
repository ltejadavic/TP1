from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from extensions import db

class Rol(db.Model):
    __tablename__ = 'Rol'
    
    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Nombre = db.Column(db.String(80), unique=True, nullable=False)

class User(UserMixin,db.Model):
    __tablename__ = 'Usuario'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    User = db.Column(db.String(80), unique=True, nullable=False)
    Password = db.Column(db.String(120), nullable=False)
    Email = db.Column(db.String(120), unique=True, nullable=False)
    RolID = db.Column(db.Integer, db.ForeignKey('Rol.ID'))
    
    rol = db.relationship('Rol', backref='users')
    
    @property
    def id(self):
        return str(self.ID)

    def __init__(self, username, password, email, hash_password=True, rol=None):
        self.User = username
        self.Password = generate_password_hash(password, method='pbkdf2:sha256') if password and hash_password else password
        self.Email = email
        self.rol = rol
        


    def check_password(self, password):
        return check_password_hash(self.Password, password)


    @classmethod
    def query_user_by_name(cls, username):
        return cls.query.filter_by(User=username).first()
