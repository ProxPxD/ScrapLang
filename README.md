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
- Languages: `--langs`, '--lang`, `-langs`, `-lang`, `-l`

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
Just like in a single translation, a common flag can be used
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
