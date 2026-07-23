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
If no language is configured, the argument is assumed to be a word.

```bash
❮ t learn -f en -t pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

### Common Language Flag
Specifying many flags is not particularly convenient, so a combined flag for languages can be used. Only the first argument is assumed to be the language to translate from. The rest is assumed to be the language(s) to translate to.
- Languages: `--langs`, `--lang`, `-langs`, `-lang`, `-l`

```bash
❮ t learn -l en pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

## Multi-Translation

### Basic
The tool allows to translate many words to many languages at once by specifying more arguments in the corresponding places.

```bash
❮ t learn teach -f en -t pl es
──── pl ──────────────────────────────────────
learn: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
teach: uczyć (verb) [impf], nauczać (verb), dydaktyka (noun), nauczyć, nauczanie, nauka, trenować, belfer, nauczyciel, nauczycielka, uczyć kogoś, wprawiać, sposobić, pouczać, uczyć się, wykładać
──── es ──────────────────────────────────────
learn: aprender (verb), estudiar (verb), enterarse (verb), averiguar, saber, enseñar, descubrir, instruir, aprendizaje, conocer, enseñanza, comprender, estudio, entender, escarmentar, practicar, enterar, ver, instruirse, percibir, ejercitar, determinar, adaptar, coger, acostumbrarse, aprehender, habituarse, aclimatarse, aprender de memoria, darse cuenta, enterarse (de), enterarse de
teach: enseñar (verb), instruir (verb), aprender (verb), educar, adiestrar, enseñanza, aprendizaje, explicar, capacitar, estudio, demostrar, dar, alfabetizar, dar clases, dar clases a, dar clases de, el profe, la profe, preparar, enterar, eseñar, aducir, manifestar, enseña, formar, acostumbrar, amaestrar, culturizar, anunciar, dar clase, hacer saber
```

### Common Language Flag Multi-Translation
Just like with a single translation, a shared flag can be used. (The output differs here to show how a single word is formatted when translated to multiple languages.)
```bash
❮ t learn -l en pl es
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
es: aprender (verb), estudiar (verb), enterarse (verb), averiguar, saber, enseñar, descubrir, instruir, aprendizaje, conocer, enseñanza, comprender, estudio, entender, escarmentar, practicar, enterar, ver, instruirse, percibir, ejercitar, determinar, adaptar, coger, acostumbrarse, aprehender, habituarse, aclimatarse, aprender de memoria, darse cuenta, enterarse (de), enterarse de
```


### Flexible
For convenience, some words may be specified implicitly, while others can be specified after a flag. As shown below, the expected order is maintained.
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
The repository stores user configuration in `resources/cong.yaml`. The configuration can be modified using the following flags:
- Setting: `--set`, `-set`, `-s`
- Adding: `--add`, `-add`, `-a`
- Deleting: `--delete`, `-delete`, `--del`, `-del`

### Configuring Languages
The most important personalization step is configuring user's preferred languages. The language codes are remembered and automatically filtered out from the words to translate.

Adding English, Polish, Spanish, and German languages:
```bash
❮ t -a lang en pl es de
```
To see all language codes, see the codes at [glosbe.com](https://glosbe.com/).

### Benefits of personalized languages

The goal of this setting is to remove the need to use flags to specify languages. The languages will be filtered out, with the first one assumed to be the so-called "from-language".
```bash
❮ t learn en pl
pl: uczyć się (verb), uczyć (verb), poznawać (verb), dowiedzieć się, nauka, usłyszeć, nauczać, nauczyć, nauczyć się, zapamiętywać, dowiedzieć, poznać, dowiadywać, zrozumieć, zapamiętać, rozeznawać, dowiadywać się
```

This also provides the freedom to specify languages in any position. All of the following are equally valid:
```bash
❮ t en learn pl
❮ t en pl learn
```

### Removing a language

```bash
❮ t -del lang de
```

### Other settings are not yet setable via CLI. Following sections will be extended later.

## Multi-From-Translation
Multi-from-translation is a concept created for a specific need. Sometimes polysemous words can be misleading, but using other languages can come to the rescue. To address this need, the tool allows multiple from-languages to be specified in an alternating pattern.

```bash
❮ t pass aprobar -f en es -t pl
pass: przechodzić (verb), podać (verb) [pf], przepustka (noun) [feminine], zdać, przełęcz, podawać, przejeżdżać, podanie, mijać, umrzeć, przebieg, podejmować, przejść, pasować, minąć, płynąć, przejechać, przekraczać, spędzać, pas, karnet, przebywać, zagranie, przekazać, podrzut, dziać się, obronienie, obronienie się, przepuścić, przemijać, zdanie, przyjąć, wydać, zdawać, dać, uchwalać, dawać, wychodzić, wyjść, uciec, wytrzymać, prześcignąć, uciekać, bilet okresowy, etap, karta wstępu, kończyć się, kłopot, ocena pozytywna, odrzucać, owijać, poddawać się, przebiegać, przedstawiać, przelatywać, przepust, przepływać, przyjmować, ruch, tracić, ustąpić, wydawać, wymijać, wynik pozytywny, zdanie egzaminu, zdany egzamin, zezwolenie, znajdować się, znikać, zwolnienie, próba, ustępować, wypad, przewyższać, włączać, odstąpić, prześcigać, przekazywać, usiłowanie, farwater, mieć powodzenie, odnieść sukces, przejście, przerzucić, zaliczyć, przepuszczać, legitymacja, zaliczenie, minięcie, stracić, wejściówka, omijać, przelecieć, uchodzić, upływać, identyfikator, przelot, przesmyk, przedostawać, zginąć, podrywać, zniknąć, bocznik, przebyć, wydalać, posłać, przebrzmieć, przeprawiać, przesuw, trója, abstrahować, poddać się
aprobar: aprobować (verb), zatwierdzać (Verb verb), pochwalać, przyjmować, uchwalać, zaliczać, zdać, pochwalić, uznawać, przyzwalać
```

In the example above, one can see that neither the English word "pass" nor the Spanish word "aprobar" gives the same answer. The English word "pass" can have many different meanings. The same is true, to a lesser extent, for the Spanish word "aprobar". Together, the translations points to the Polish word "zdać", which is the first common match. A back-translation confirms this, as this word translates back to its English and Spanish counterparts as its primary meaning:
```bash
❮ t zdać pl en es
en: pass (verb), turn over (verb), to graduate (verb), to pass, to realize, qualify
es: aprobar (verb), pasar (verb)
```

### Multi-From- and To-Translation
All multi-modes can be used at once. In that case, the alternative syntax can be seen in the words. Groups will be printed to make the output easier to read:
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

## Shared Parts Syntax
Sometimes there is a need to compare words that are related to one another. The tool provides a syntax for specifying such words without the need to reapet oneself.

### Replacemnt
This syntax is especially useful when one wants to compare translations that differ depending on a certain varying element, such as prefixes in some languages.
```bash
❮ t [ver,be]sprechen de en
versprechen: promise (noun) [verb], pledge (verb), misspeak (verb), fluff, hope, expect, word, undertaking, mispronounce, keep, assure, fulfil, guarantee, back, engagement, meet, plight, abide, secure, accomplish, to pledge, to promise, execute, achieve, protect, affirm, warrant, insure, underwrite, vouch, certify, safeguard, ensure, perform, exercise, observe, abide by, cover, have sexual intercourse, share a bed, vow, undertake, prometer, to reduce
besprechen: discuss (verb), review (verb), talk over (verb), bespeak, agitate, discussion, consult, stir up, talk together, debate, to confer, to criticize, to discuss, to review, to talk over, arrange, criticize, arouse, incite, talk about, to critique, to talk, record, to arrange, to consult somebody, to drive away, to record, to sweep out, to talk about, to talk something over
```

There may be a need to check the unprefixed word as well. For example, the German word "sprechen" is valid on its own. To support this case, an empty position can be added:

```bash
❮ t [ver,be,]sprechen de en
versprechen: promise (noun) [verb], pledge (verb), misspeak (verb), fluff, hope, expect, word, undertaking, mispronounce, keep, assure, fulfil, guarantee, back, engagement, meet, plight, abide, secure, accomplish, to pledge, to promise, execute, achieve, protect, affirm, warrant, insure, underwrite, vouch, certify, safeguard, ensure, perform, exercise, observe, abide by, cover, have sexual intercourse, share a bed, vow, undertake, prometer, to reduce
besprechen: discuss (verb), review (verb), talk over (verb), bespeak, agitate, discussion, consult, stir up, talk together, debate, to confer, to criticize, to discuss, to review, to talk over, arrange, criticize, arouse, incite, talk about, to critique, to talk, record, to arrange, to consult somebody, to drive away, to record, to sweep out, to talk about, to talk something over
sprechen: speak (verb), talk (verb), say (verb), tell, converse, express, pronounce, argue, utter, recite, story, propose, distinguish, discriminate, manage, carry through, dispose of, speaking respectfully, talking, to articulate, to broadcast, to converse, to pronounce, to put up, to recite, to rehearse, to say, to see, to speak, to talk, chat, utterance, past, history, pretext, eat, taste, be noticed, be recognized, be seen, be visible, refer to, speak of, to negotiate, discourse, administer, chatter, natter, ... says, according to ..., speak (irr.), speak to, talk to, to be noticed, to be recognized, to be seen, to be visible, to call, to have a talk, to open one's mouth, to put into words, to speak {spoke, spoken} (about), to tell a story
```

Those produce the same words, just in a different order:
```bash
❮ t [ver,,be]sprechen de en
❮ t [,ver,be]sprechen de en
```

## Empty Replacement Syntax
It is worth noting that if a single element is specified in the brackets "[X]", it is treated as if there were a comma in front of it: "[,X]". This way, one may easier remove certain parts and learn the entire consecutive language logic. This syntax can also be nested.

```bash
❮ t nation[al[ize[d]]] en es 
nation: nación (noun) [feminine], pueblo (noun) [masculine], estado (noun) [masculine], país, región, patria, lugar, población, gente, provincia, el pueblo, la nación, república, tierra, tribu, urbe, poblado, aldea, ciudad, paraje, villa, pago
national: nacional (adjective) [masculine, feminine], ciudadano (noun) [masculine], súbdito (adjective noun) [masculine], público, estatal, criollo, el periódico nacional, la competencia nacional, nacionales, patrio, doméstico, gentilicio, national, ciudadana, súbdita
nationalize: nacionalizar (verb), estatificar, socializar (verb)
nationalized: nacionalizada (adjective) [feminine], nacionalizado (adjective) [masculine]
```

### Post-Segmental Replacement - Removal

Another way to use the syntax is to specify the replacement after writing the word. This feature allows changes to be introduced without going back to add brackets,keeping the writing flow continuous. The "/" syntax indicates the removal of previous characters. To remove multiple characters, either specify the number (e.g., "/3") or repeat the symbol (e.g., "///").

```bash
❮ t national/2 en pl
national: narodowy (adjective) [masculine], krajowy (adjective) [masculine], narodowościowy (adjective), obywatel, państwowy, narodowa, narodowe, ogólnonarodowy, nacjonalny, ogólnokrajowy, krajoznawczy
nation: naród (noun) [masculine], nacja (noun) [feminine], lud (noun) [masculine], państwo, kraj, stan, nawa państwowa, Naród
```
### Post-Segmental Replacement - True Replacement
To add a part after the removal, it has to be written after the "/" operator. This is especially useful when a suffix removes a previously existing ending.

```bash
❮ t baking/3e en pl
baking: pieczenie (noun) [neuter], wypiekanie (noun), piekarniany (adjective), wypiekowy, piekący, Pieczenie, piekarnictwo, piekarstwo, wypalanie, wypał, wypieczenie, wypiek
bake: piec (verb) [impf], wypalać (verb), upiec (verb), zapiekać, wypiekać, gotować się, jarać, piec się, pieczenie w piekarniku, wypalanie w piecu, zapiekanka, prażyć, dopalić, smazyć, pięć
```

### Post-Segmental Replacement - Replacement boundary
When the replacement occurs before the end and all words share the same ending, a "." can be used to indicate the position.

```bash
❮ t be//ver.sprechen de en
besprechen: discuss (verb), review (verb), talk over (verb), bespeak, agitate, discussion, consult, stir up, talk together, debate, to confer, to criticize, to discuss, to review, to talk over, arrange, criticize, arouse, incite, talk about, to critique, to talk, record, to arrange, to consult somebody, to drive away, to record, to sweep out, to talk about, to talk something over
versprechen: promise (noun) [verb], pledge (verb), misspeak (verb), fluff, hope, expect, word, undertaking, mispronounce, keep, assure, fulfil, guarantee, back, engagement, meet, plight, abide, secure, accomplish, to pledge, to promise, execute, achieve, protect, affirm, warrant, insure, underwrite, vouch, certify, safeguard, ensure, perform, exercise, observe, abide by, cover, have sexual intercourse, share a bed, vow, undertake, prometer, to reduce
```

## Non-Translation Modes

### Overview
The tool provides a way to gather general information about words by adding an overview flag.
- overview: `--overview`, `-overview`, `-o`

Specifying a to-language is not required, but it is allowed.

```bash
❮ t learn en -o
learn: 
meanings:
  • [PoS: Verb, third-person singular simple present: learns, present participle: learning, simple past and past participle: learned or learnt]
  pronunciation:
    - Received Pronunciation: /lɜːn/
    - General American: /lɝn/
    - dialectal: /lɑː(ɹ)n/
    - Early Modern: /lɛːrn/, /lɛrn/
    etymology:
      - from Middle English lernen (“to learn", also, "to teach"”)
      - from Old English leornian (“to learn", rarely also, "to teach”)
      - from Proto-West Germanic *liʀnōn
      - from Proto-Germanic *lizaną
      - from Proto-Indo-European *(le-)lóys-e, stative from the root *leys- (“track, furrow, trace, trail”)
  • [PoS: Noun, plural: learns]
  pronunciation:
    - Received Pronunciation: /lɜːn/
    - General American: /lɝn/
    - dialectal: /lɑː(ɹ)n/
    - Early Modern: /lɛːrn/, /lɛrn/
    etymology:
      - from Middle English lernen (“to learn", also, "to teach"”)
      - from Old English leornian (“to learn", rarely also, "to teach”)
      - from Proto-West Germanic *liʀnōn
      - from Proto-Germanic *lizaną
      - from Proto-Indo-European *(le-)lóys-e, stative from the root *leys- (“track, furrow, trace, trail”)
```

### Translated Word Overview
To address the need to translate a word and get additional data immediately, one can use "t" after the `-o` flag to request an overview of the to-lang word, as in `-ot`. The tool will perform the translation and retrieve the overview right afterward.

```bash
❮ t teach en pl -ot
pl: uczyć (verb) [impf], nauczać (verb), dydaktyka (noun), nauczyć, nauczanie, nauka, trenować, belfer, nauczyciel, nauczycielka, uczyć kogoś, wprawiać, sposobić, pouczać, uczyć się, wykładać
meanings:
  • /ˈu.t͡ʂɘt͡ɕ/ [PoS: Verb, gender: impf]
    etymology:
      - Inherited from Old Polish uczyć
      - from Proto-Slavic *učiti
```

### Grammar

To gather short grammatical information about the words, use the grammar flag:
- gather grammar data: `--grammar`, `-grammar`, `-g`

```bash
❮ t learn en -g 
learn: 
  - learn, learns, learning, learned, learnt
  - s, ing
```

The same rule as for the overview applies to to-language data gathering.


### Inflection
Inflection gathering is used to display inflection tables for verbs and nouns in langauges that provide them. If an inflection table is not available, the tool falls back to the grammar flag.

How to display the inflection of the German word for "dog":
```bash
❮ t dog en de -it
╭───┬────────────┬───────┬─────┬───────────────────────┬─────┬────────╮
│ 0 │ nominative │ ein   │ der │ Hund                  │ die │ Hunde  │
│ 1 │ genitive   │ eines │ des │ Hundes, Hunds         │ der │ Hunde  │
│ 2 │ dative     │ einem │ dem │ Hund, Hunde (archaic) │ den │ Hunden │
│ 3 │ accusative │ einen │ den │ Hund                  │ die │ Hunde  │
╰───┴────────────┴───────┴─────┴───────────────────────┴─────┴────────╯
de: Hund (noun) [masculine], Rüde (noun) [masculine], verfolgen (verb) [v], Hündin, jagen, Köter, Töle, Hunde, hetzen, Haushund, plagen, fixieren, befestigen, festmachen, Dackel, Klammer, Pleite, Schreckschraube, Vogelscheuche, Schurke, sichern, Schuft, an den Fersen bleiben, Bock, Knagge, Mitnehmer, schreckschraube, Wauwau, Auflagebock, Feuerbock, Klampe, Mitnehmerstift, Schabracke, Nocke, Hunt, nachspüren, Hund -es, Hunde-, Klaue, Klemme
```

### Mixed Modes

These additional flags can be combined in any sequence, for example `-igt` or `-git`. However, it is recommended to place the side "t" at the end, as putting it elsewhere may lead to confusing collisions, such as "-to", which is a flag on its own. Other combinations are still allowed for convenience, such as "-otg".
