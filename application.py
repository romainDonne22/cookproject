#################################################################################
############ BACKEND GERE avec FastAPI et SQLAlchemy pour la partie bdd #########
#################################################################################
# ouvrer l'environenement Poetry
# lancer l'application avec la commande suivante dans un terminal :
# uvicorn application:app --reload
# aller sur http://127.0.0.1:8000/docs#/default
#################################################################################

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from databases import Database

DATABASE_URL = "sqlite:///recipes.db"

database = Database(DATABASE_URL)
metadata = MetaData()

recipes_table = Table(
    "recipes",
    metadata,
    Column("id", Integer, primary_key=True, index=True, autoincrement=True),
    Column("name", String, index=True),
    Column("description", String, index=True),
    Column("rating", Float, index=True),
    Column("ingredients", String, index=True),
)

engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

Base = declarative_base()

class RecipeInDB(Base):
    __tablename__ = "recipes"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    description = Column(String, index=True)
    rating = Column(Float, index=True)
    ingredients = Column(String, index=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

class Recipe(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    rating: Optional[float] = None
    ingredients: List[str] = []

    class Config:
        orm_mode = True

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

### route pour récupérer toutes les recettes dans un id intervalle
@app.get("/recipes", response_model=List[Recipe])
async def get_recipes(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    recipes = db.query(RecipeInDB).offset(skip).limit(limit).all()
    return [Recipe(
        id=recipe.id,
        name=recipe.name,
        description=recipe.description,
        rating=recipe.rating,
        ingredients=recipe.ingredients.split(",") if recipe.ingredients else []
    ) for recipe in recipes]

### route pour ajouter une recette
@app.post("/recipes", response_model=Recipe)
async def add_recipe(recipe: Recipe, db: Session = Depends(get_db)):
    db_recipe = RecipeInDB(
        name=recipe.name,
        description=recipe.description,
        rating=recipe.rating,
        ingredients=",".join(recipe.ingredients),
    )
    db.add(db_recipe)
    db.commit()
    db.refresh(db_recipe)
    return Recipe(
        id=db_recipe.id,
        name=db_recipe.name,
        description=db_recipe.description,
        rating=db_recipe.rating,
        ingredients=db_recipe.ingredients.split(",") if db_recipe.ingredients else []
    )

### route pour récupérer une recette depuis son id
@app.get("/recipes/{recipe_id}", response_model=Recipe)
async def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(RecipeInDB).filter(RecipeInDB.id == recipe_id).first()
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return Recipe(
        id=recipe.id,
        name=recipe.name,
        description=recipe.description,
        rating=recipe.rating,
        ingredients=recipe.ingredients.split(",") if recipe.ingredients else []
    )

### route pour mettre à jour une recette depuis son id
@app.put("/recipes/{recipe_id}", response_model=Recipe)
async def update_recipe(recipe_id: int, updated_recipe: Recipe, db: Session = Depends(get_db)):
    recipe = db.query(RecipeInDB).filter(RecipeInDB.id == recipe_id).first()
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    recipe.name = updated_recipe.name
    recipe.description = updated_recipe.description
    recipe.rating = updated_recipe.rating
    recipe.ingredients = ",".join(updated_recipe.ingredients)
    db.commit()
    db.refresh(recipe)
    return Recipe(
        id=recipe.id,
        name=recipe.name,
        description=recipe.description,
        rating=recipe.rating,
        ingredients=recipe.ingredients.split(",") if recipe.ingredients else []
    )

### route pour supprimer une recette depuis son id
@app.delete("/recipes/{recipe_id}", response_model=Recipe)
async def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(RecipeInDB).filter(RecipeInDB.id == recipe_id).first()
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    db.delete(recipe)
    db.commit()
    return Recipe(
        id=recipe.id,
        name=recipe.name,
        description=recipe.description,
        rating=recipe.rating,
        ingredients=recipe.ingredients.split(",") if recipe.ingredients else []
    )

# Route pour page hello qui affiche "Hello World"
@app.get("/hello", response_model=str)
async def hello_world():
    return "Hello World"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True, log_level="debug")