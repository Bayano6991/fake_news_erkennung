# fake_news_erkennung
Web-Mining Projekt zur Erkennung von Fake News aus diversen Informationsquellen

### Beschreibung
Das Ziel des Projektes ist es, Fake News aus verschiedenen Datenquellen anhand ML Models zu erkennen. Die Nachrichten stammen aus diversen Bereichen wie Politik, Wirtschaft, Kultur, Wissenschaft und Sport. Zum Training der Daten (Texten) wurde mehrere Modellen verwendet wie *RoBERTa*, *distil-RoBERTa* oder *deberta*. Als Datenzsatz zum Training der Modelle wurden LIAR und Fever verwendet. Die Voraussagen sind leider nicht immer zuverlässig, denn die Trainingsdaten wurden nicht genug bzw. das Model zum Training hatte nicht genug Parameter.

### Vorgehensweise

#### Sammlung der Daten
*python <name>_collector.py* <br> 'name' könnte reddit, youtube oder telegram sein je nachdem aus welcher Quelle man die Daten erhalten möchte. Dafür sollte man die Umgebungsveriable Datei (.env) mit entsprechenden Informationen ausfüllen.

#### Trainieren der Modelle
  *python train_<suffix>.py*. <br><br>Es gibt mehrere Modelltrainier. Der *train_fever.py* im Vergleich zu anderen verwendet nicht LIAR sondern Fever (https://fever.ai/dataset/fever.html oder https://huggingface.co/datasets/fever/fever) als Datensatz zum Trainieren.


#### Vorhersagen
  Um die Vorhersagen zu machen, könnte man eigentlich per Text oder mit der Angabe einer csv Datei machen. Die Daten aus Quellen werden als JSON gespeichert. Um sie als csv zu konvertieren, sollte man dias Befehl *python data_converter.py* der Ordner 'collectors' ausführen. Erst dann kann man den Pfad dieser Datei als Argument angeben.
  
  Beispiel<br>
     *python predict_fever.py --csv ../collectors/csv/telegram.csv --out predictions/predictions_telegram.csv* <br>
     *python predict_fever.py --text "Germany is in central Europe"*
