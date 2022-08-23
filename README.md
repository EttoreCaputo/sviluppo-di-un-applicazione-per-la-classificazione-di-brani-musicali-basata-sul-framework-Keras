# Sviluppo di un applicazione per la classificazione di brani musicali basata sul framework Keras
Lavoro di Tesi di laurea in Ingegneria Informatica.

Il codice di questo repository contiene un applicazione di deep learning per la classificazione di generi musicali, in particolare: metal e rock ma è possibile allenare la rete con qualsiasi altri. In questo caso le tracce audio provengono dal dataset **GTZAN**.

Nell'applicazione sono presenti tecniche di pre-elaborazione dei dati effettuate mediante l'uso della libreria **Librosa**, per l'estrazione dei **MFCCs** dalle singole tracce audio.

La rete utilizzata è di tipo **RNN**.

Il codice in questione è stato eseguito su *Google Colab*. Contiene un piccolo script shell per l'installazione di *Kaggle* e il download del dataset. Al fine di eseguire lo script senza intoppi, è opportuno creare un account Kaggle ed ottenere così il file token (kaggle.json nel codice) che bisogna inserire nella directory principale del server di Google Colab, solo in questo modo sarà possibile accedere alle API di Kaggle ed effettuare il download del dataset.

Il miglior risultato ottenuto per quanto riguarda l'accuratezza raggiunta dal modello sul test set è **94.5%**
