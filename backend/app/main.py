
"""
συνδέει όλα τα routes
Όλη η πραγματική λειτουργικότητα βρίσκεται στο api/routes.py και στα services
Αν ο backend σηκωθεί σωστά, το /health επιστρέφει ok.
"""

from fastapi import FastAPI

# Εισάγουμε το κεντρικό router του API,
# το οποίο περιλαμβάνει όλα τα επιμέρους routes
# (auth, admin, datasets, federated jobs, consent, audit κ.λπ.)
from app.api.routes import router as api_router

# Δημιουργία FastAPI application
# Το title χρησιμοποιείται:
# - στο OpenAPI schema
# - στο Swagger UI (/docs)
app = FastAPI(title="BC-FL Platform API")

# Εγγραφή (mount) του κεντρικού router στο app
# Όλα τα endpoints θα είναι προσβάσιμα κάτω από τα prefixes
# που ορίζονται μέσα στο api/routes.py (π.χ. /api/v1/...)
app.include_router(api_router)


@app.get("/")
def root():
    """
    Root endpoint.

    Χρησιμοποιείται κυρίως:
    - για γρήγορο έλεγχο ότι ο backend server τρέχει
    - ως απλό sanity check (π.χ. μέσω browser ή curl)

    Δεν απαιτεί authentication.
    """
    return {"message": "BC-FL Platform API"}


@app.get("/health")
def health():
    """
    Health check endpoint.

    Χρησιμοποιείται από:
    - Docker / orchestration
    - monitoring εργαλεία
    - γρήγορο έλεγχο διαθεσιμότητας

    Δεν ελέγχει database ή blockchain·
    απλώς επιβεβαιώνει ότι το FastAPI process είναι ενεργό.
    """
    return {"status": "ok"}
