from extensions import db
from .User import User

class Dataset(db.Model):
    __tablename__ = 'Dataset'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Nombre = db.Column(db.String(255), nullable=False)
    Descripción = db.Column(db.Text, nullable=True)
    UsuarioID = db.Column(db.Integer, db.ForeignKey('Usuario.ID'), nullable=False)
    Fecha = db.Column(db.DateTime, nullable=False)
    FilePath = db.Column(db.String(255), nullable=False)

    # Relación con el usuario
    usuario = db.relationship('User', backref=db.backref('datasets', lazy=True))

    def __init__(self, Nombre, Descripción, UsuarioID, Fecha, FilePath):
        self.Nombre = Nombre
        self.Descripción = Descripción
        self.UsuarioID = UsuarioID
        self.Fecha = Fecha
        self.FilePath = FilePath