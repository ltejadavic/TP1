from extensions import db
from .User import User

class Preppred(db.Model):
    __tablename__ = 'Preppred'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    UsuarioID = db.Column(db.Integer, db.ForeignKey('Usuario.ID'), nullable=False)  # Aquí es donde necesitas hacer el cambio
    Fecha = db.Column(db.DateTime, default=db.func.current_timestamp())
    Comentario = db.Column(db.Text, nullable=True)
    FilePath = db.Column(db.String(255), nullable=False)
    Nombre = db.Column(db.String(255), nullable=False)

    # Relaciones
    usuario = db.relationship('User', backref=db.backref('preppreds', lazy=True))

    def __init__(self, UsuarioID, Fecha,Comentario, FilePath, Nombre):
        self.UsuarioID = UsuarioID
        self.Fecha = Fecha
        self.Comentario = Comentario
        self.FilePath = FilePath
        self.Nombre = Nombre
