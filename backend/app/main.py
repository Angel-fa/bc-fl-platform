# συνδέει όλα τα routes

from fastapi import FastAPI

from app.api.routes import router as api_router

app = FastAPI(title="BC-FL Platform API")


app.include_router(api_router)


@app.get("/")
def root():
    return {"message": "BC-FL Platform API"} # έλεγχος ότι ο backend server τρέχει


@app.get("/health")
def health():
    return {"status": "ok"} #   Health check -> αν FastAPI  είναι ενεργό.
