from extensions import db
from .User import User
from .Dataset import Dataset

class Preprocesamiento(db.Model):
    __tablename__ = 'Preprocesamiento'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    DatasetID = db.Column(db.Integer, db.ForeignKey('Dataset.ID'), nullable=False)
    UsuarioID = db.Column(db.Integer, db.ForeignKey('Usuario.ID'), nullable=False)
    Fecha = db.Column(db.DateTime, nullable=False)
    Comentario = db.Column(db.Text, nullable=True)
    FilePath = db.Column(db.String(255), nullable=True)
    X_trainPath = db.Column(db.String(255), nullable=True)
    X_testPath = db.Column(db.String(255), nullable=True)
    y_trainPath = db.Column(db.String(255), nullable=True)
    y_testPath = db.Column(db.String(255), nullable=True)

    # Relaciones
    dataset = db.relationship('Dataset', backref=db.backref('preprocesamientos', lazy=True))
    usuario = db.relationship('User', backref=db.backref('preprocesamientos', lazy=True))

    def __init__(self, DatasetID, UsuarioID, Fecha, Comentario, FilePath, X_trainPath, X_testPath, y_trainPath, y_testPath):
        self.DatasetID = DatasetID
        self.UsuarioID = UsuarioID
        self.Fecha = Fecha
        self.Comentario = Comentario
        self.FilePath = FilePath
        self.X_trainPath = X_trainPath
        self.X_testPath = X_testPath
        self.y_trainPath = y_trainPath
        self.y_testPath = y_testPath
