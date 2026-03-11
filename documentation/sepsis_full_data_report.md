# Report Dataset Sepsis Full (Allineato Huang et al. 2025)

Il dataset è stato ricostruito per riflettere fedelmente il benchmark Sepsis-3 di riferimento, espandendo il set di feature da 22 a **55 variabili cliniche**.

## 1. Statistiche Generali
- **Soggetti Unici**: 62.325 (ICU stays filtrati per età > 18 e durata > 12h)
- **Risoluzione Temporale**: 4 ore per bin
- **Finestra di Osservazione**: 48 ore totali (12 timestep)
- **Tasso di Mortalità**: 10.06% (Bilanciamento decisamente migliore rispetto ai subset precedenti)
- **Copertura Dati (Sparsity)**: ~24% di valori osservati (tipico di MIMIC-IV)

## 2. Categorie di Variabili Selezionate

### Vitals & Hemodynamics (12 feature)
Monitoraggio continuo dei segni vitali e della pressione arteriosa.
- *Inclusi*: Heart Rate, SBP, DBP, MBP, Resp Rate, SpO2, Temp, CVP, PA Pressures (systolic, diastolic, mean), Cardiac Index.

### Laboratory Values (30 feature)
Esami biochimici, gas analisi ed ematologia. Fondamentali per il calcolo del SOFA score e per identificare disfunzioni d'organo.
- *Elettroliti*: Potassio, Sodio, Cloro, Magnesio, Calcio.
- *Disfunzione Organica*: Creatinina (Reni), Bilirubina (Fegato), Piastrine (Coagulazione), Troponina (Cuore), Lattato (Shock/Metabolismo).
- *Infiammazione*: WBC, CRP.
- *Gas Analisi*: pH, FiO2, PaO2, PaCO2, HCO3, Total CO2.
- *Altro*: Glucosio, Emoglobina, Ematocrito, RBC, PT, PTT, INR, Urea Nitrogen.

### Neurological & Respiratory (10 feature)
Stato di coscienza e supporto meccanico.
- *Neuro*: GCS (Glasgow Coma Scale), Richmond-RAS Scale.
- *Respiratory Meta*: Oxygen Flow, PEEP, Tidal Volume, Minute Volume, Airway Pressures (mean, peak, plateau).

### Interventi Clinici (3 feature)
- **Urine Output**: Volume di urina nelle 4 ore.
- **Norepi Equiv**: Dose totale equivalente di vasopressori (standardizzata).
- **Vent Status**: Flag binario per ventilazione meccanica.

## 3. Logica di Preprocessing Applicata
1. **Clinical Outlier Removal**: Rimozione di valori fisiologicamente impossibili (es. HR > 250, MBP > 200).
2. **FiO2 Correction**: Normalizzazione dei valori tra 21 e 100%.
3. **Binning & Aggregation**: Media dei valori all'interno delle finestre di 4 ore.
4. **Standardization**: Z-score normalization basata esclusivamente sulle statistiche del training set.
