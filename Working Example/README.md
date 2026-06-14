# Fraud Detection Working Example

Dieses Verzeichnis enthaelt ein vollstaendiges Beispiel fuer Training,
Hyperparameterauswahl und Auswertung eines kleinen neuronalen Netzes.

## Voraussetzungen

- Python 3.10 oder neuer
- Die Pakete aus `requirements.txt`

Installation:

```powershell
python -m pip install -r requirements.txt
```

## Ausfuehren

Die Befehle muessen im Projektverzeichnis ausgefuehrt werden:

```powershell
python run_fraud_project.py
```

Das Skript vergleicht verschiedene Netzwerkarchitekturen, waehlt die beste
Konfiguration anhand der Validierungsdaten und wertet sie danach einmalig auf
den Testdaten aus.

Zum Vergleich verschiedener L2-Regularisierungsstaerken:

```powershell
python compare_l2.py
```

Dieses Skript speichert das ausgewaehlte Modell als
`best_validation_model.pt`.

## Dateien

- `transactions.zip`: Datensatz mit `transactions.csv`
- `module1_data.py`: Laden, Aufteilen und Standardisieren der Daten
- `minimal_neural_network.py`: Netzwerk ohne L2-Regularisierung
- `minimal_neural_network_l2.py`: Netzwerk mit L2-Regularisierung
- `module3_evaluation.py`: Berechnung der Evaluationsmetriken
- `run_fraud_project.py`: Architekturvergleich und abschliessende Auswertung
- `compare_l2.py`: Vergleich der L2-Staerken und Speichern des besten Modells
