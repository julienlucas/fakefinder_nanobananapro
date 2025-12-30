# Extension Chrome FakeFinder

## Installation

1. Installer les dépendances (si nécessaire):
```bash
cd /Users/Julien/Desktop/fakefinder
uv sync
```

2. Démarrer l'API (⚠️ **depuis le répertoire racine du projet**, pas depuis `chrome-extension`):
```bash
cd /Users/Julien/Desktop/fakefinder
uv run uvicorn inference_api:app --reload --port 8000
```

**Important**: Assurez-vous d'être dans `/Users/Julien/Desktop/fakefinder` et non dans `chrome-extension` avant de lancer la commande.

3. Charger l'extension dans un navigateur Chromium:

   **Étapes détaillées:**
   1. Ouvrir votre navigateur (Chrome, Edge, Brave, etc.)
   2. Aller à la page des extensions:
      - **Chrome**: Tapez `chrome://extensions/` dans la barre d'adresse
      - **Edge**: Tapez `edge://extensions/` dans la barre d'adresse
      - **Brave**: Tapez `brave://extensions/` dans la barre d'adresse
      - **Opera**: Tapez `opera://extensions/` dans la barre d'adresse
   3. Activer le **"Mode développeur"** (toggle en haut à droite)
   4. Cliquer sur **"Charger l'extension non empaquetée"** (ou "Load unpacked")
   5. Naviguer et sélectionner le dossier `/Users/Julien/Desktop/fakefinder/chrome-extension`
   6. L'extension devrait apparaître dans la liste et une icône dans la barre d'outils

4. Utiliser l'extension:
   - Cliquer sur l'icône de l'extension
   - Glisser-déposer ou sélectionner une image
   - Voir le résultat de l'analyse

## Notes

- L'API doit être démarrée sur `http://localhost:8000`
- Pour la production, modifier `API_URL` dans `popup.js` et ajouter les permissions CORS appropriées

