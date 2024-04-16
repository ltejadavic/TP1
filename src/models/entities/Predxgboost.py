from extensions import db
from datetime import datetime

class Predxgboost(db.Model):
    __tablename__ = 'Predxgboost'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UsuarioID = db.Column(db.Integer, db.ForeignKey('Usuario.ID'), nullable=False)
    ModeloID = db.Column(db.Integer, db.ForeignKey('Trainxgboost.ID'), nullable=False)
    FilePath = db.Column(db.String(255), nullable=False)
    Fecha = db.Column(db.DateTime, default=datetime.utcnow)
    Accuracy = db.Column(db.Float, nullable=False)
    NombrePrediccion = db.Column(db.String(255), nullable=False)
    PredFilePath = db.Column(db.String(255), nullable=False)

    # Relaciones
    usuario = db.relationship('User', backref=db.backref('predxgboosts', lazy=True))
    modelo = db.relationship('Trainxgboost', backref=db.backref('predxgboosts', lazy=True))

    def __init__(self, UsuarioID, ModeloID, FilePath, Fecha, Accuracy, NombrePrediccion, PredFilePath):
        self.UsuarioID = UsuarioID
        self.ModeloID = ModeloID
        self.FilePath = FilePath
        self.Fecha = Fecha
        self.Accuracy = Accuracy
        self.NombrePrediccion = NombrePrediccion
        self.PredFilePath = PredFilePath
