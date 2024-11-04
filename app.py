from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recipes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(200), nullable=True)
    rating = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<Recipe {self.name}>'

@app.route('/recipes', methods=['GET'])
def get_recipes():
    recipes = Recipe.query.all()
    return jsonify([{'id': recipe.id, 'name': recipe.name, 'description': recipe.description, 'rating': recipe.rating} for recipe in recipes])

@app.route('/recipes', methods=['POST'])
def add_recipe():
    data = request.get_json()
    new_recipe = Recipe(name=data['name'], description=data.get('description'), rating=data.get('rating'))
    db.session.add(new_recipe)
    db.session.commit()
    return jsonify({'message': 'Recipe added successfully'}), 201

@app.route('/recipes/<int:id>', methods=['GET'])
def get_recipe(id):
    recipe = Recipe.query.get_or_404(id)
    return jsonify({'id': recipe.id, 'name': recipe.name, 'description': recipe.description, 'rating': recipe.rating})

@app.route('/recipes/<int:id>', methods=['PUT'])
def update_recipe(id):
    data = request.get_json()
    recipe = Recipe.query.get_or_404(id)
    recipe.name = data['name']
    recipe.description = data.get('description')
    recipe.rating = data.get('rating')
    db.session.commit()
    return jsonify({'message': 'Recipe updated successfully'})

@app.route('/recipes/<int:id>', methods=['DELETE'])
def delete_recipe(id):
    recipe = Recipe.query.get_or_404(id)
    db.session.delete(recipe)
    db.session.commit()
    return jsonify({'message': 'Recipe deleted successfully'})

# Ajouter une route pour afficher "Hello World"
@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello World"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)