# ScrapLang Overview
ScrapLang is a CLI tool to quickly and comfortably translate words and gather linguistical data through web scrapping for displayal.

# Usage

## Single Translation

### Explicit

Arguments can be specified using flags:
- language(s) to translate from: `--from-langs`, `--from-lang`, `--from`, `-from`, `-f`
- langauge(s) to translate to: `--to-langs`, `--to-lang`, `--to`, `-to`, `-t`
- word(s) to translate: `--word`, `--word`, `-w`

```bash
❮ t -f en -t pl -w learn
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

### Default Words
If no language is configured, the argument is assumed to be a word

```bash
❮ t learn -f en -t pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

### Common Language Flag
Specifying many flags is not exactly the most comfortable, so a joined flag for languages can be used. Only the first argument is assumed to be the language to translate from. The rest is assumed to be the language(s) to translate to
- Languages: `--langs`, `--lang`, `-langs`, `-lang`, `-l`

```bash
❮ t learn -l en pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

## Multi Translation

### Basic
The tool allows to translate many words to many languages at once by specifying more arguments in the corresponding places

```bash
❮ t learn teach -f en -t pl es
──── pl ──────────────────────────────────────
learn: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
teach: uczyć (verb) [impf], nauczać (verb), dydaktyka (noun), nauczyć, nauczanie, nauka, trenować, belfer, nauczyciel, nauczycielka, uczyć kogoś, wprawiać, sposobić, pouczać, uczyć się, wykładać
──── es ──────────────────────────────────────
learn: aprender (verb), estudiar (verb), enterarse (verb), averiguar, saber, enseñar, descubrir, instruir, aprendizaje, conocer, enseñanza, comprender, estudio, entender, escarmentar, practicar, enterar, ver, instruirse, percibir, ejercitar, determinar, adaptar, coger, acostumbrarse, aprehender, habituarse, aclimatarse, aprender de memoria, darse cuenta, enterarse (de), enterarse de
teach: enseñar (verb), instruir (verb), aprender (verb), educar, adiestrar, enseñanza, aprendizaje, explicar, capacitar, estudio, demostrar, dar, alfabetizar, dar clases, dar clases a, dar clases de, el profe, la profe, preparar, enterar, eseñar, aducir, manifestar, enseña, formar, acostumbrar, amaestrar, culturizar, anunciar, dar clase, hacer saber
```

### Common Language Flag Multi Translation
Just like in a single translation, a common flag can be used. (Here the output differs to show how one-word translated to multiple languages is formatted)
```bash
❮ t learn -l en pl es
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
es: aprender (verb), estudiar (verb), enterarse (verb), averiguar, saber, enseñar, descubrir, instruir, aprendizaje, conocer, enseñanza, comprender, estudio, entender, escarmentar, practicar, enterar, ver, instruirse, percibir, ejercitar, determinar, adaptar, coger, acostumbrarse, aprehender, habituarse, aclimatarse, aprender de memoria, darse cuenta, enterarse (de), enterarse de
```


### Flexible
For the comfort of specifying arguments whenever one likes, some words may be specified implicitly while some after a flag. As it can be seen below, the expected order is maintained
```bash
❮ t learn -l en pl es -w teach
──── pl ──────────────────────────────────────
learn: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
teach: uczyć (verb) [impf], nauczać (verb), dydaktyka (noun), nauczyć, nauczanie, nauka, trenować, belfer, nauczyciel, nauczycielka, uczyć kogoś, wprawiać, sposobić, pouczać, uczyć się, wykładać
──── es ──────────────────────────────────────
learn: aprender (verb), estudiar (verb), enterarse (verb), averiguar, saber, enseñar, descubrir, instruir, aprendizaje, conocer, enseñanza, comprender, estudio, entender, escarmentar, practicar, enterar, ver, instruirse, percibir, ejercitar, determinar, adaptar, coger, acostumbrarse, aprehender, habituarse, aclimatarse, aprender de memoria, darse cuenta, enterarse (de), enterarse de
teach: enseñar (verb), instruir (verb), aprender (verb), educar, adiestrar, enseñanza, aprendizaje, explicar, capacitar, estudio, demostrar, dar, alfabetizar, dar clases, dar clases a, dar clases de, el profe, la profe, preparar, enterar, eseñar, aducir, manifestar, enseña, formar, acostumbrar, amaestrar, culturizar, anunciar, dar clase, hacer saber
```

## Personalization
The repository stores the user configuration in `resources/cong.yaml`. To manipulate the configuration, use configuration flags:
- Setting: `--set`, `-set`, `-s`
- Adding: `--add`, `-add`, `-a`
- Deleting: `--delete`, `-delete`, `--del`, `-del`

### Configuring Languages
The most important personalization is setting up user languages. The language codes will be remembered and filtered out out of the words.

Adding English, Polish, Spanish and German languages:
```bash
❮ t -a lang en pl es de
```
To see all language code see the codes at [glosbe.com](https://glosbe.com/).

### Benefits of personalized languages

The goal of this setting is to remove the need to use flags to specify the languages. Languages will be filtered out with the first one assumed to be the so-called from-language.
```bash
❮ t learn en pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

This also guarantees a freedom to specify languages language in any position. All the following are equally valid:
```bash
❮ t en learn pl
❮ t en pl learn
```

### Removing a language

```bash
❮ t -del lang de
```

### Other settings are not yet setable via CLI. Following sections will be extended later

## Multi From Translation
Multi from translation is a concept created for a specific need. Sometimes polysemic words can deceive, but taking other languages can come to rescue. For that needs, the tool allows to specify many from-languages in an alternating pattern

```bash
❮ t pass aprobar -f en es -t pl
pass: przechodzić (verb), podać (verb) [pf], przepustka (noun) [feminine], zdać, przełęcz, podawać, przejeżdżać, podanie, mijać, umrzeć, przebieg, podejmować, przejść, pasować, minąć, płynąć, przejechać, przekraczać, spędzać, pas, karnet, przebywać, zagranie, przekazać, podrzut, dziać się, obronienie, obronienie się, przepuścić, przemijać, zdanie, przyjąć, wydać, zdawać, dać, uchwalać, dawać, wychodzić, wyjść, uciec, wytrzymać, prześcignąć, uciekać, bilet okresowy, etap, karta wstępu, kończyć się, kłopot, ocena pozytywna, odrzucać, owijać, poddawać się, przebiegać, przedstawiać, przelatywać, przepust, przepływać, przyjmować, ruch, tracić, ustąpić, wydawać, wymijać, wynik pozytywny, zdanie egzaminu, zdany egzamin, zezwolenie, znajdować się, znikać, zwolnienie, próba, ustępować, wypad, przewyższać, włączać, odstąpić, prześcigać, przekazywać, usiłowanie, farwater, mieć powodzenie, odnieść sukces, przejście, przerzucić, zaliczyć, przepuszczać, legitymacja, zaliczenie, minięcie, stracić, wejściówka, omijać, przelecieć, uchodzić, upływać, identyfikator, przelot, przesmyk, przedostawać, zginąć, podrywać, zniknąć, bocznik, przebyć, wydalać, posłać, przebrzmieć, przeprawiać, przesuw, trója, abstrahować, poddać się
aprobar: aprobować (verb), zatwierdzać (Verb verb), pochwalać, przyjmować, uchwalać, zaliczać, zdać, pochwalić, uznawać, przyzwalać
```

In the example above, one can see that neither English "pass" nor Spanish "aprobar" gives the same answer. English "pass" can have quite many meanings. The same, but to a lesser extent, is true for the Spanish "aprobar". Together the common translation hints to Polish "zdać" which is the first common match. A back-translation confirms it as this word translate to its English and Spanish counterparts as a primary meaning:
```bash
❮ t zdać pl en es
en: pass (verb), turn over (verb), to graduate (verb), to pass, to realize, qualify
es: aprobar (verb), pasar (verb)
```

### Multi From and To Translation
All multi-modes can be used at once. Groups will be printed to ease reading the output
```bash
❮ t pass aprobar know conocer -f en es -t pl de
━━━━ pl ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
──── pass·aprobar ────────────────────────────
pass: przechodzić (verb), podać (verb) [pf], przepustka (noun) [feminine], zdać, przełęcz, podawać, przejeżdżać, podanie, mijać, umrzeć, przebieg, podejmować, przejść, pasować, minąć, płynąć, przejechać, przekraczać, spędzać, pas, karnet, przebywać, zagranie, przekazać, podrzut, dziać się, obronienie, obronienie się, przepuścić, przemijać, zdanie, przyjąć, wydać, zdawać, dać, uchwalać, dawać, wychodzić, wyjść, uciec, wytrzymać, prześcignąć, uciekać, bilet okresowy, etap, karta wstępu, kończyć się, kłopot, ocena pozytywna, odrzucać, owijać, poddawać się, przebiegać, przedstawiać, przelatywać, przepust, przepływać, przyjmować, ruch, tracić, ustąpić, wydawać, wymijać, wynik pozytywny, zdanie egzaminu, zdany egzamin, zezwolenie, znajdować się, znikać, zwolnienie, próba, ustępować, wypad, przewyższać, włączać, odstąpić, prześcigać, przekazywać, usiłowanie, farwater, mieć powodzenie, odnieść sukces, przejście, przerzucić, zaliczyć, przepuszczać, legitymacja, zaliczenie, minięcie, stracić, wejściówka, omijać, przelecieć, uchodzić, upływać, identyfikator, przelot, przesmyk, przedostawać, zginąć, podrywać, zniknąć, bocznik, przebyć, wydalać, posłać, przebrzmieć, przeprawiać, przesuw, trója, abstrahować, poddać się
aprobar: aprobować (verb), zatwierdzać (Verb verb), pochwalać, przyjmować, uchwalać, zaliczać, zdać, pochwalić, uznawać, przyzwalać
──── know·conocer ────────────────────────────
know: wiedzieć (verb) [impf], znać (verb) [Modal], umieć (Modal), znać się, poznawać, doświadczać, poznać, zaznawać, dowiedzieć, rozumieć, odróżniać, rozróżniać, znać jak swoje dziesięć palców
conocer: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
━━━━ de ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
──── pass·aprobar ────────────────────────────
pass: vorbeigehen (verb), passieren (verb), Pass (noun) [masculine], vergehen, bestehen, verbringen, überholen, vorbeifahren, verfließen, vorübergehen, Ausweis, geben, passen, durchlaufen, reichen, übergeben, Passierschein, absolvieren, vorüberfahren, verstreichen, verabschieden, weitergeben, übergehen, durchgehen, verlaufen, abgeben, Arbeitsgang, Durchlauf, Bergpass, fällen, übertreffen, ablaufen, fahren, gehen, Durchgang, herreichen, hindurchgehen, Abgabe, abspielen, annehmen, übersteigen, Zuspiel, sterben, vorbeikommen, erreichen, versterben, verscheiden, eintrittskarte, überreichen, überbieten, Gebirgspass, hinter sich lassen, reisepass, Bestehen, hindurchfahren, stattfinden, Fahrrinne, Paß, hinreichen, ablegen, erlassen, gelingen, genehmigen, vertreiben, überqueren, dahinscheiden, einhändigen, Hinscheiden, Freikarte, entschlafen, verrinnen, Base-on-Balls, Pässe, verfliegen, schaffen, Abspiel, Annäherungsversuch, Bestehen einer Prüfung, Eintritt, Engpaß, Entschuldigung, Erlaubnisschein, Freifahrschein, Joch, Legitimation, Notlage, Passen, Urlaubsschein, Vorbeiflug, Weitergabe, bestandene Prüfung, durchkommen, eintreten, gelten, hereinkommen, laufen, machen, vorbeiführen, vorbeilaufen, vorbeiziehen, vorüberziehen, „Bestanden“, entlangfahren, fassen, Monatskarte, Fahrwasser, zubringen, Stoß, gutheißen, hinausgehen, Passstraße, entlanggehen, gewähren, spenden, Ausgeherlaubnis, genehmigung, tragen, Ausfall, erteilen, ausgehen, abnippeln, draufgehen, verrecken, heimgehen, schenken, krepieren, hinausfahren, verleben, abkratzen, anvertrauen, überschreiten, überantworten, erzeugen, fließen, angeben, Durchkommen, anbieten, verwirklichen, Ausgang, gestatten, hervorbringen, Zug, realisieren, ausrücken, Eintrittskarte, verabreichen, Erfolg haben, das Zeitliche segnen, den Geist aufgeben, den Löffel abgeben, ins Gras beißen, pass, zu weit fahren, Dauerkarte, Reisepass, Urlaub, Engpass, Kaliber, ausscheiden, Zeitkarte, zuspielen, Passweg, Zeitfahrkarte, schicken, Bewegung, Aminosalicylsäure, versagen, Höhepunkt, Vorlage, senden, helfen, Aufschlag, Öffnung, Krise, Angabe, Steig, Raupe, Durchlassschein, Klause, unterstützen, Durchfahrt, Fahrwasser -s, Gipfel, Passe, Reisepass, Passierschein, Rolle, Scharte, Versuch, bekannt sein, bestehen (Prüfung), den Arsch zukneifen, den Weg alles Irdischen gehen, die Augen für immer schließen, einen guten Zug haben, fördern, hops gehen, in die ewigen Jagdgründe eingehen, kritischer Punkt, kritisches Stadium, para-Aminosalicylsäure PAS, sein Leben aushauchen, verläuft, vorbei sein, über den Jordan gehen, über die Klinge springen
aprobar: billigen (verb), gutheißen (verb), genehmigen (verb), goutieren, zustimmen, bestehen, sanktionieren, approbieren, annehmen, bewilligen, beistimmen, bestehen lassen, bestätigen, durchkommen, entlasten, verabschieden, abnehmen, einverstanden, loben, absolvieren, einwilligen, beipflichten, autorisieren, zulassen, erlauben
──── know·conocer ────────────────────────────
know: wissen (verb) [v], kennen (verb) [v], können (verb), erkennen, verstehen, lernen, auskennen, fühlen, Bescheid, denken, begreifen, spüren, bemerken, herausbekommen, unterscheiden, folgern, auseinanderhalten, Bescheid wissen, sich auskennen, sich auskennen in, sich bewusst sein, bekannt sein mit, wissen, kennen
conocer: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
```
