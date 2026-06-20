# Institutional Portfolio Sandbox & Quantitative Risk Engine

## Panoramica
Questa applicazione rappresenta un motore di analisi quantitativa per la costruzione di portafogli e la valutazione del rischio. È stata progettata per simulare flussi di lavoro tipici di divisioni di analisi dati e ricerca statistica presso istituzioni finanziarie.

Il progetto integra tecniche di:
* **Analisi di serie storiche:** estrazione dati tramite yfinance e pulizia matriciale.
* **Econometria del rischio:** calcolo di metriche non-Gaussiane (Sortino, CVaR 95%).
* **Ottimizzazione:** simulazioni Monte Carlo per la frontiera efficiente.

## Prova l'applicazione (Live Demo)
Puoi testare l'interfaccia interattiva direttamente nel tuo browser:
 **[CLICCA QUI PER LA LIVE DEMO](INSERISCI_QUI_IL_LINK_CHE_TI_DARA_STREAMLIT)**

## Struttura del progetto
- `/core`: Contiene il motore di calcolo (`quant_engine.py`), dove risiede la logica quantitativa.
- `app.py`: Interfaccia utente (Streamlit) che gestisce l'input dell'utente e la visualizzazione dei risultati.
- `requirements.txt`: Dipendenze necessarie per l'esecuzione.

## Come eseguire il progetto localmente
Se preferisci eseguire il codice sul tuo ambiente locale:
1. Clona il repository: `git clone https://github.com/TUO_NOME_UTENTE/Institutional-Portfolio-Sandbox.git`
2. Installa le dipendenze: `pip install -r requirements.txt`
3. Avvia l'app: `streamlit run app.py`
