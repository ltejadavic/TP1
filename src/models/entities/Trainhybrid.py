from extensions import db
from .User import User
from .Dataset import Dataset

class Trainhybrid(db.Model):
    __tablename__ = 'Trainhybrid'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UsuarioID = db.Column(db.Integer, db.ForeignKey('Usuario.ID'), nullable=False)
    DatasetID = db.Column(db.Integer, db.ForeignKey('Dataset.ID'), nullable=False)
    Accuracy = db.Column(db.Float, nullable=False)
    Recall = db.Column(db.Float, nullable=False)
    F1Score = db.Column(db.Float, nullable=False)
    ModeloxgbPath = db.Column(db.String(255), nullable=False)
    ModelornnPath = db.Column(db.String(255), nullable=False)
    ModeloNombre = db.Column(db.String(255), nullable=False)
    Fecha = db.Column(db.DateTime, nullable=False)

    # Relaciones
    usuario = db.relationship('User', backref=db.backref('trainhyb', lazy=True))
    dataset = db.relationship('Dataset', backref=db.backref('trainhyb', lazy=True))

    def __init__(self, UsuarioID, DatasetID, Accuracy, Recall, F1Score, ModeloxgbPath, ModelornnPath, ModeloNombre, Fecha):
        self.UsuarioID = UsuarioID
        self.DatasetID = DatasetID
        self.Accuracy = Accuracy
        self.Recall = Recall
        self.F1Score = F1Score
        self.ModeloxgbPath = ModeloxgbPath
        self.ModelornnPath = ModelornnPath
        self.ModeloNombre = ModeloNombre
        self.Fecha = Fecha