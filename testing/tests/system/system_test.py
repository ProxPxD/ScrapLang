from __future__ import annotations

import shlex
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from itertools import product, permutations
from pathlib import Path
from typing import Iterable, Any, Sequence, Collection, Optional

import pydash as _
import pytest
import yaml
from box import Box
from more_itertools import windowed
from pydash import chain as c

from src.app_managing import AppMgr
from src.context_domain import assume, indirect, gather_data, infervia, groupby
from src.exceptions import InvalidExecution
from testing.core import TCG
from testing.core.mocking import mocked_scrap, CallCollector, PageNotFound
from testing.core.utils import remove_color

SYSTEM_PATH = Path(__file__).parent
TMP_DIR = SYSTEM_PATH / 'tmp'
TEST_CONF = TMP_DIR / 'test_conf.yaml'

TMP_DIR.mkdir(exist_ok=True)



@dataclass
class InputCase:
    input: Iterable[str] | str
    tags: Iterable[str] = field(default_factory=list)
    conf: Box = field(default_factory=Box)
    context: dict[str, Any] = field(default_factory=dict)
    output: str = ''
    exception: type[BaseException] | Sequence[type[BaseException]] = None
    skip_mocking: bool = False

    replacement: dict = field(default_factory=dict)

IC = InputCase

@dataclass
class TC:
    descr: str
    input: Iterable[InputCase | str] | InputCase | str
    tags: Iterable[str] = field(default_factory=list)
    conf: dict[str, Any] = field(default_factory=Box)
    context: dict[str, Any] = field(default_factory=dict)
    output: str = ''
    exception: type[BaseException] | Sequence[type[BaseException]] = None
    color: bool = False
    skip_mocking: bool = False

    replacement: dict = field(default_factory=dict)


@dataclass
class Tc:
    descr: str
    tags: Sequence[str]

    input: str
    conf: dict
    context: dict
    output: str
    exception: type[BaseException] | Sequence[type[BaseException]] = None
    color: bool = False
    skip_mocking: bool = False


class SystemTCG(TCG):

    color_confs = ('no', 'blue', dict(main='blue'), dict(main='yellow', pronunciation=[0, 255, 0]))
    mappings = dict(
        simple_single=dict(eo={'Cx': 'Ĉ', 'Gx': 'Ĝ', 'Hx': 'Ĥ', 'Jx': 'Ĵ', 'Sx': 'Ŝ', 'Ux': 'Ŭ', 'cx': 'ĉ', 'gx': 'ĝ', 'hx': 'ĥ', 'jx': 'ĵ', 'sx': 'ŝ', 'ux': 'ŭ'}),
        ordered=dict(ru=[{'lu': 'лю'}, {'u': 'у'}]),
        regex=dict(ru=[{'l([aeuo])': 'łi\1', 'li': 'łi'}, {'[ji]a': 'я', '[ji]e': 'е', '[ji]o': 'ё', '[ji]u': 'ю', 'j[''q]': 'й'}]),
    )
    single_confs = [conf for conf in [
        *[dict(indirect=indirect_val) for indirect_val in indirect],
        *[dict(assume=assume_val) for assume_val in assume],
        *[dict(gather_data=gather_data_val) for gather_data_val in gather_data],
        *[dict(infervia=infervia_val) for infervia_val in infervia],
        *[dict(groupby=groupby_val) for groupby_val in groupby],
        *[dict(color=color_conf) for color_conf in color_confs],
        *[dict(mappings=mapping) for mapping in mappings.values()]
    ] if next(iter(conf.values())) != 'conf']

    perm_base = Box({
        'from_langs': ('en',),
        'to_langs': ('pl', 'de'),
        'words': ('water', 'bass'),
    })
    multi_perm = Box({
        'langs': (*perm_base.from_langs, *perm_base.to_langs),
        'args': (*perm_base.from_langs, *perm_base.to_langs, *perm_base.words),
        **perm_base,
    })

    @classmethod
    def generate_tcs(cls) -> list:
        bs = "\\"
        return [
            TC(
                descr='Exact all-flag configless call',
                tags=['all-flag', 'position', 'permutation'],
                input=_.map_(permutations({'-w Frau', '-f de', '-t pl'}, 3), ' '.join),
                context={
                    'from_langs': ('de',),
                    'words': ['Frau'],
                    'to_langs': ['pl'],
                },
                output=(frau_translation := '''
                            Frau: kobieta (noun) [feminine], żona (noun) [feminine], pani (noun) [abbreviation], małżonka, babka, mężatka, lady, Pani, kobièta, dama, WPani, facetka, kobita, kobitka, panna, ona, niewiasta, baba, samica, białogłowa, dupa, babsko, czyściocha, jejmość, p.
                ''')
            ),
            TC(
                descr=f'Conf Loading',
                tags=['conf', 'load'],
                input=[
                    IC(
                        input=f'-h',
                        context={f'_conf.{next(iter(conf.keys()))}': next(iter(conf.values()))},
                        conf=conf,
                    ) for conf in cls.single_confs
                ],
            ),
            TC(
                descr=f'Conf Setting',
                tags=['conf', 'set'],
                input=[
                    IC(
                        tags=[f'{next(conf.keys())}/{next(conf.values())}'],
                        input=f'--set {next(conf.keys())} {next(conf.values())}',
                        context={f'_conf.{next(conf.keys())}': next(conf.values())},
                    ) for conf in cls.single_confs if not isinstance(conf, (dict, list))
                ],
            ),
            TC(
                descr='Assume sunny resolution',
                tags={'assume'},
                input=[
                    'es de -w en de orden --assume lang',  # lang
                    'en de orden --assume word',  # word
                    'es de -w en de orden',  # Default
                    IC(
                        tags={'to-lang/arg', 'to-lang/flag', 'to-lang/dual'},
                        input='crime en fr -t sv ar',
                        context={
                            'words': ['crime'],
                            'from_langs': ('en',),
                            'to_langs': ['fr', 'sv', 'ar'],
                        },
                        skip_mocking=True,
                        output=None,
                    ),
                ],
                context={
                    'from_langs': ('es',),
                    'words': ['en', 'de', 'orden'],
                    'to_langs': ['de'],
                },
                conf=(just_langs_es_de_pl_en_conf := Box({
                    'langs': ['es', 'de', 'pl', 'en', 'fr'],
                })),
                output='''
                    en: in (adposition), an (adposition), auf (adposition), bei, über, anderswo, zu, nach, im, binnen, neben, verstehen, für, mit, unter, innerhalb, b., um, während, zwischen, inmitten, drin, von, gestellt, gegen, je, pro, at, à, nahe bei, um zu, en, hinein, mit Hilfe von, nahe
                    de: von (adposition), aus (adposition), vor (adposition), der, über, nach, ab, einig, des, auf, mit, an, als, seit, alt, lösen, abgehen, weggehen, oberirdisch, sekkieren, ein düsteres Bild zeichnen, verantwortlich zeichnen, von ... her, um, v., für, in, zu, sein, bei, müssen, Brenner, losmachen, wegtun, abnabeln, abmachen, wegmachen, Blessur, ab-, aus ... heraus, deren, dessen, ent-, sich lösen, von ... an, weg-, wegen, de, Angelegenheit, innen, entbehren, bewundern, -s, unzählbares Substantiv
                    orden: Ordnung (noun) [masculine], Befehl (noun) [feminine], Reihenfolge (noun) [masculine], Orden, Kommando, Auftrag, Anordnung, Anweisung, Gebot, Bestellung, ordnung, Order, Geheiß, Verfügung, Regierung, Behörde, Leitung, Regieren, befehlen, Reihe, Vorstand, verordnung, Vorschrift, bestellen, serie, Bereich, Erlass, Geschäftsordnung, Instruktion, Komplex, Königswürde, Ordentlichkeit, Rang, Rangordnung, Recht, Systematik, Säulenordnung, Veranlassung, Weihe, Weihen, Ordnungsrelation, Aneinanderreihung, Abbuchungsauftrag, Verkettung, Organisation, Richtlinie, Sinn, zeitliche Ordnung, Verordnung, Weisung, Dienst, Gesetz, System, Sortierung, Diktat, Klasse, Ordo, Serie, Kategorie, Gesetzesentwurf, Gesetzgebung, Gerichtsurteil, Methode, Orden und Ehrenzeichen, Urteil, orden und ehrenzeichen#verdienstauszeichnungen
                '''
            ),
            TC(
                descr='Assume rainy resolution',
                input={
                    'es pl de --assume lang'
                },
                output='No word specified!',
                conf=just_langs_es_de_pl_en_conf,
            ),
            TC(
                descr='Single translation of any arg placement resolution',
                tags={'permutation', 'position', 'single/from-langs', 'single/to-langs', 'single/words'},
                input={
                    'Herr de pl',
                    'de Herr pl',
                    'de pl Herr',
                },
                context={
                    'from_langs': ('de',),
                    'words': ['Herr'],
                    'to_langs': ['pl'],
                },
                conf=(base_langs_es_de_pl_en_conf := Box(dict(**just_langs_es_de_pl_en_conf, assume='lang'))),
            ),
            TC(
                descr='Multiword from single lang to multilang translation of any arg placement resolution',
                tags={'permutation', 'position', 'multi/words', 'multi/to-langs', 'single/from-lang'},
                input={
                    ' '.join(perm)
                    for perm
                    in permutations(cls.multi_perm.args, len(cls.multi_perm.args))
                    if all(i < j for group in (cls.multi_perm.langs, cls.multi_perm.words) for i, j in windowed(map(perm.index, group), 2))
                },
                context={
                    'from_langs': ('en',),
                    'words': ['water', 'bass'],
                    'to_langs': ['pl', 'de'],
                },
                conf=base_langs_es_de_pl_en_conf,
            ),
            TC(
                descr='Main Replacements',
                tags=set(),
                input={
                    'es pl orden conocer <WORD_FLAG> en de',
                    'pl orden conocer -w en de <FROM_LANGS> es',
                    'es orden conocer -w en de <TO_LANG> pl',
                },
                replacement={
                    'WORD_FLAG': ['--word', '--words', '-word', '-words'],
                    'FROM_LANGS': ['--from-lang', '--from', '-from'],
                    'TO_LANG':   ['--to-lang', '--to-langs', '--to', '-to'],
                },
                context={
                    'words': ['orden', 'conocer', 'en', 'de'],
                    'from_langs': ('es',),
                    'to_langs': ['pl'],
                },
                conf=base_langs_es_de_pl_en_conf,
            ),
            TC(
              descr='Option separation',
                tags=['separate/argument', 'separate/option'],
                input='en de -- -wards',
                context={
                    'words': ['-wards'],
                    'from_langs': ('en',),
                    'to_langs': ['de'],
                },
                conf=base_langs_es_de_pl_en_conf,
                skip_mocking=True,
            ),
            TC(
                descr='No break space separation',
                input='en pl bass water',
                tags=['break-space'],
                context={
                    'from_langs': ('en',),
                    'to_langs': ['pl'],
                    'words': ['bass', 'water']
                },
                conf=base_langs_es_de_pl_en_conf
            ),
            TC(
                descr='Uniqness',
                tags={'repeated', 'duplicated', 'uniq'},
                input=[
                    IC(
                        tags={'uniq/word'},
                        input='Frau Frau de pl',
                        context={'words': ['Frau']},
                    ),
                    IC(
                        tags={'uniq/to-lang'},
                        input='Frau de pl pl',
                        context={'to_langs': ['pl']},
                    ),
                ],
                conf=base_langs_es_de_pl_en_conf,
            ),
            # TODO: extend cutting and modyfy outstemming by introducing another operator working as post-removal wihtout replacement
            TC(
                descr='Outstemming',
                tags={'outstem'},
                skip_mocking=True,
                input=[
                    # Outstem/Bracket
                    IC(
                        tags={'bracket/one-option'},
                        input='pl en spot[y]kać',
                        context={'words': ['spotkać', 'spotykać']}
                    ),
                    IC(
                        tags={'bracket/trailing', 'sep/after'},
                        input=[f'pl en spot[y{sep}]kać' for sep in ',|'],
                        context={'words': ['spotykać', 'spotkać']}
                    ),
                    IC(
                        tags={'bracket/trailing', 'sep/before'},
                        input=[f'pl en spot[{sep}y]kać' for sep in ',|'],
                        context={'words': ['spotkać', 'spotykać']}
                    ),
                    IC(
                        tags={'bracket/many-options'},
                        input=[f'pl en [nad{sep}do{sep}prze]robić' for sep in ',|'],
                        context={'words': ['nadrobić', 'dorobić', 'przerobić']}
                    ),
                    IC(
                        tags={'bracket/nested'},
                        input=f'en pl nation[al[ize]]',
                        context={'words': ['nation', 'national', 'nationalize']}
                    ),
                    IC(
                        tags={'bracket/two'},
                        input='en pl [pre]heat[ed]',
                        context={'words': ['heat', 'heated', 'preheat', 'preheated']}
                    ),
                    IC(
                        tags={'bracket/separated', 'outstem/join'},
                        input='en pl [to ]water',
                        context={'words': ['water', 'to water']},
                    ),
                    IC(
                        tags={'bracket/two', 'bracket/repeated'},
                        input='[a]bb[a]',
                        context={'words': ['bb', 'bba', 'abb', 'abba']}
                    ),
                    IC(
                        tags={'bracket/two', 'bracket/repeated', 'outestem/remove/empty'},
                        input='de [teil][nehmen]',
                        context={'words': ['nehmen', 'teil', 'teilnehmen']},
                    ),
                    IC(
                        tags={'bracket/two', 'bracket/repeated', 'outestem/trim/space', 'space'},
                        input='en "[password] [manager]"',
                        context={'words': ['manager', 'password', 'password manager']},
                    ),

                    # Outstem/Cut
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/1', 'seq'},
                        input='eo viro/ino/iĝi',
                        context={'words': ['viro', 'virino', 'viriniĝi']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/3', 'seq'},
                        input='eo viriniĝi///o///o',
                        context={'words': ['viriniĝi', 'virino', 'viro']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/3', 'cut/size/2', 'cut/empty/placetaker', 'seq'},
                        input='en nationalize///_//',
                        context={'words': ['nationalize', 'national', 'nation']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/1', 'cut/empty/placeholder', 'seq'},
                        input='en nation_/al_/ize',
                        context={'words': ['nation', 'national', 'nationalize']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/3', 'cut/size/2', 'cut/number', 'seq'},
                        input='en nationalize/3/2',
                        context={'words': ['nationalize', 'national', 'nation']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/1', 'cut/size/1', 'cut/end'},
                        input='pl przy/e.łożyć',
                        context={'words': ['przyłożyć', 'przełożyć']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/1', 'cut/size/1', 'cut/end', 'cut/bracket', 'bracket/cut'},
                        input='pl przy/e[d].stawić',
                        context={'words': ['przystawić', 'przestawić', 'przedstawić']}
                    ),
                    IC(
                        tags={'cut/post', 'cut/times/2', 'cut/size/3', 'cut/size/2', 'cut/end', 'cut/number', 'cut/many'},
                        input='eo _/tra,el,en,al.iri',
                        context={'words': ['iri', 'trairi', 'eliri', 'eniri', 'aliri']},
                    ),
                    # IC(  # TODO handle?
                    #     tags={'cut/post', 'cut/times/2', 'cut/size/3', 'cut/size/2', 'cut/end', 'cut/number', 'cut/many'},
                    #     input='pl do,s,po\\chować',
                    #     context={'words': ['iri', 'trairi', 'eliri', 'eniri', 'aliri']},
                    # ),
                    # IC(  # TODO: Think if this should be imlpemented at all (current solution: [[hin]zu]geben
                    #     tags={'cut/post', 'cut/times/2', 'cut/size/3', 'cut/size/2', 'cut/end', 'cut/number', 'cut/many'},
                    #     input='de hin/3.zu/2.geben',
                    #     context={'words': []}  # TODO: fill
                    # ),
                ],
                conf={**base_langs_es_de_pl_en_conf, 'langs': ['eo', 'es', 'en', 'pl', 'de']},
            ),
            TC(
                descr='Inflection',
                tags={'inflection'},
                input=[
                    IC(
                        tags={'inflection', 'nontranslate', 'permutation', 'position'},
                        input=_.map_(permutations({'Frau', 'de', '-i'}, 3), ' '.join),
                        context=(frau_infl_context := {
                            'from_langs': ('de',),
                            'to_langs': [],
                            'inflection': True,
                            'words': ['Frau']
                        }),
                        output=(frau_inflection := '''
                            ╭───┬────────────┬───────┬─────┬──────┬─────┬────────╮
                            │ 0 │ nominative │ eine  │ die │ Frau │ die │ Frauen │
                            │ 1 │ genitive   │ einer │ der │ Frau │ der │ Frauen │
                            │ 2 │ dative     │ einer │ der │ Frau │ den │ Frauen │
                            │ 3 │ accusative │ eine  │ die │ Frau │ die │ Frauen │
                            ╰───┴────────────┴───────┴─────┴──────┴─────┴────────╯
                        ''')
                    ),
                    IC(
                        input='Frau de <INFL_FLAG>',
                        replacement={'INFL_FLAG': ['--inflection', '--infl', '-infl']},
                        context=frau_infl_context,
                    ),
                    IC(
                        input='слать ru -i',
                        context={'inflection': True},
                    ),
                ],
                conf=(assume_langs_pl_de_en_es_ru := Box({
                    'langs': ['pl', 'de', 'en', 'es', 'ru'],
                })),
            ),
            TC(
                descr='Definition',
                tags={'definition'},
                input=[
                    IC(
                        tags={'nontranslate', 'definition', 'permutation', 'position'},
                        input=_.map_(permutations({'Frau', 'de', '-d'}, 3), ' '.join),
                        context=(frau_def_context := {
                            'from_langs': ('de',),
                            'definition': True,
                            'words': ['Frau']
                        }),
                        output=(frau_definition := '''
                            Definitions of "Frau":
                            - Weib (derb)
                            - Weibsstück (abwertend) (derb)
                            - Eine verheiratete Frau.
                            - Anrede für einen erwachsenen, weiblichen Menschen.
                            - engl. Anrede, die keinen Unterschied zwischen verheiratet und unverheiratet macht
                            - Slang
                            - unhöflich
                            - höfl. für Frau oder Kinder
                            - Erwachsene, menschliche Angehörige des Geschlechts, das Eizellen produziert und Kinder gebärt.
    
                        '''),
                    ),
                    IC(
                        input='Frau <DEF_FLAG> de',
                        replacement={
                            'DEF_FLAG': ['--definitions', '--definition', '-definitions', '-definition', '--def', '-def']},
                        context=frau_def_context,
                    ),
                ],
                conf=assume_langs_pl_de_en_es_ru,
            ),
            TC(
                descr='Overview',
                tags={'overview', 'wiktio', 'nontranslate'},
                input=[
                    IC(
                        tags={'nontranslate', 'permutation', 'position'},
                        input=_.map_(permutations({'Frau', 'de', '-o'}, 3), ' '.join),
                        context=(frau_overview_context := {
                            'from_langs': ('de',),
                            'wiktio': True,
                            'words': ['Frau']
                        }),
                        output=(frau_overview := '''
                            Frau: 
                            meanings:
                              • /fʁaʊ̯/ [PoS: Noun, gender: f, genitive: Frau, plural: Frauen, diminutive: Fräulein n or Frauchen n]
                                etymology:
                                  - from Middle High German vrouwe
                                  - from Old High German frouwa (“mistress”)
                                  - from Proto-West Germanic *frauwjā
                                  - from Proto-Germanic *frawjǭ, a feminine form of *frawjô (“lord”), giving Old English frēa (“lord, king; God, Christ; husband”), frēo (“woman”)
                                  - from Proto-Indo-European *proHwo-, a derivation from *per- (“to go forward”)
                        '''),
                    ),
                    IC(
                        input='Frau <OVERVIEW_FLAG> de',
                        replacement={
                            'OVERVIEW_FLAG': ['--wiktio', '-wiktio', '--overview', '-overview', '-o',]},
                        context=frau_overview_context,
                    ),
                ],
                conf=assume_langs_pl_de_en_es_ru,
            ),
            TC(
                descr='Reverse',
                tags=set(),
                input=[
                    IC(
                        tags={'permutation', 'position'},
                        input={
                            'pl es conocer -r',
                            'pl es -r conocer',
                            'pl -r es conocer',
                        },
                        context=(conocer_context := {
                            'from_langs': ('es',),
                            'to_langs': ['pl'],
                            'words': ['conocer'],
                        }),
                    ),
                    IC(
                        tags={'from-conf'},
                        input='-r conocer',
                        conf=Box({
                            'langs': ['pl', 'es'],
                            'assume': 'lang',
                        }),
                        context=conocer_context,
                    )
                ],
                conf=base_langs_es_de_pl_en_conf,
            ),
            TC(
                descr='GroupBy',
                tags=set(),
                input=[
                    IC(
                        tags={'flag/groupby/lang'},
                        input='es pl de -w conocer orden --groupby lang',
                        output='''
                            ──── pl ──────────────────────────────────────
                            conocer: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                            orden: porządek (noun) [masculine], rząd (noun) [masculine], rozkaz (noun) [masculine], zakon, zamówienie, polecenie, nakaz, kolejność, order, zarząd, zarządzenie, instrukcja, dyspozycja, kategoria, komenda, odznaczenie, rozporządzenie, szyk, układ, zalecenie, zamówić, wyrok, ustawa, zamawiać, przykazanie, zlecenie, relacja porządkowa, relacja porządku, ład, wskazówka, ustawodawstwo, prawodawstwo, konta, odpisu, uporządkowanie
                            ──── de ──────────────────────────────────────
                            conocer: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
                            orden: Ordnung (noun) [masculine], Befehl (noun) [feminine], Reihenfolge (noun) [masculine], Orden, Kommando, Auftrag, Anordnung, Anweisung, Gebot, Bestellung, ordnung, Order, Geheiß, Verfügung, Regierung, Behörde, Leitung, Regieren, befehlen, Reihe, Vorstand, verordnung, Vorschrift, bestellen, serie, Bereich, Erlass, Geschäftsordnung, Instruktion, Komplex, Königswürde, Ordentlichkeit, Rang, Rangordnung, Recht, Systematik, Säulenordnung, Veranlassung, Weihe, Weihen, Ordnungsrelation, Aneinanderreihung, Abbuchungsauftrag, Verkettung, Organisation, Richtlinie, Sinn, zeitliche Ordnung, Verordnung, Weisung, Dienst, Gesetz, System, Sortierung, Diktat, Klasse, Ordo, Serie, Kategorie, Gesetzesentwurf, Gesetzgebung, Gerichtsurteil, Methode, Orden und Ehrenzeichen, Urteil, orden und ehrenzeichen#verdienstauszeichnungen
                        '''
                    ),
                    IC(
                        tags={'flag/groupby/word'},
                        input='es pl de -w conocer orden -by word',
                        output='''
                            ──── conocer ─────────────────────────────────
                            pl: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                            de: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
                            ──── orden ───────────────────────────────────
                            pl: porządek (noun) [masculine], rząd (noun) [masculine], rozkaz (noun) [masculine], zakon, zamówienie, polecenie, nakaz, kolejność, order, zarząd, zarządzenie, instrukcja, dyspozycja, kategoria, komenda, odznaczenie, rozporządzenie, szyk, układ, zalecenie, zamówić, wyrok, ustawa, zamawiać, przykazanie, zlecenie, relacja porządkowa, relacja porządku, ład, wskazówka, ustawodawstwo, prawodawstwo, konta, odpisu, uporządkowanie
                            de: Ordnung (noun) [masculine], Befehl (noun) [feminine], Reihenfolge (noun) [masculine], Orden, Kommando, Auftrag, Anordnung, Anweisung, Gebot, Bestellung, ordnung, Order, Geheiß, Verfügung, Regierung, Behörde, Leitung, Regieren, befehlen, Reihe, Vorstand, verordnung, Vorschrift, bestellen, serie, Bereich, Erlass, Geschäftsordnung, Instruktion, Komplex, Königswürde, Ordentlichkeit, Rang, Rangordnung, Recht, Systematik, Säulenordnung, Veranlassung, Weihe, Weihen, Ordnungsrelation, Aneinanderreihung, Abbuchungsauftrag, Verkettung, Organisation, Richtlinie, Sinn, zeitliche Ordnung, Verordnung, Weisung, Dienst, Gesetz, System, Sortierung, Diktat, Klasse, Ordo, Serie, Kategorie, Gesetzesentwurf, Gesetzgebung, Gerichtsurteil, Methode, Orden und Ehrenzeichen, Urteil, orden und ehrenzeichen#verdienstauszeichnungen
                        '''
                    ),
                    IC(
                        tags={'flag/groupby/word', 'ungroup'},
                        input='es pl -w conocer orden --groupby word',
                        output='''
                            conocer: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                            orden: porządek (noun) [masculine], rząd (noun) [masculine], rozkaz (noun) [masculine], zakon, zamówienie, polecenie, nakaz, kolejność, order, zarząd, zarządzenie, instrukcja, dyspozycja, kategoria, komenda, odznaczenie, rozporządzenie, szyk, układ, zalecenie, zamówić, wyrok, ustawa, zamawiać, przykazanie, zlecenie, relacja porządkowa, relacja porządku, ład, wskazówka, ustawodawstwo, prawodawstwo, konta, odpisu, uporządkowanie
                        '''
                    ),
                    IC(
                        tags={'flag/groupby/lang', 'ungroup'},
                        input='es pl -w conocer orden --groupby lang',
                        output='''
                            conocer: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                            orden: porządek (noun) [masculine], rząd (noun) [masculine], rozkaz (noun) [masculine], zakon, zamówienie, polecenie, nakaz, kolejność, order, zarząd, zarządzenie, instrukcja, dyspozycja, kategoria, komenda, odznaczenie, rozporządzenie, szyk, układ, zalecenie, zamówić, wyrok, ustawa, zamawiać, przykazanie, zlecenie, relacja porządkowa, relacja porządku, ład, wskazówka, ustawodawstwo, prawodawstwo, konta, odpisu, uporządkowanie
                        '''
                    ),
                    IC(
                        tags={'flag/groupby/lang', 'ungroup'},
                        input='es pl de -w conocer --groupby lang',
                        output='''
                            pl: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                            de: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
                        ''',
                    ),
                    IC(
                        tags={'flag/groupby/word', 'ungroup'},
                        input='es pl de -w conocer --groupby word',
                        output='''
                            pl: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry                                                                                                       
                            de: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen  
                        '''
                    ),
                ],
                conf=base_langs_es_de_pl_en_conf,
            ),
            TC(
                descr='Multi from-langs',
                tags={'multi/from-langs'},
                input=[
                    IC(
                        tags=set(),
                        input='kobieta Frau -f pl de -t en',
                        context={
                            'words': ['kobieta', 'Frau'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en'],
                        },
                        output='''
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                        ''',
                    ),
                    IC(
                        tags={'multi/to-langs', 'groupby/lang'},
                        input='kobieta Frau -f pl de -t en es --groupby=lang',
                        context={
                            'words': ['kobieta', 'Frau'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en', 'es'],
                        },
                        output='''
                            ━━━━ en ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ━━━━ es ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            kobieta: mujer (noun) [feminine], hembra (noun) [feminine], mujeres (noun) [feminine], femenino, señor, tía, mina, papel de la mujer, sexo femenino
                            Frau: mujer (noun) [feminine], esposa (noun) [feminine], señora (noun) [feminine], Señora, doña, marida, hembra, mujeres, Sra, fémina, individua, máquina, pariente, señor, dama, casar, cónyuge, persona, mina, galla, Sra., ella, sexo femenino, señorita, sra., tía, usted
                        ''',
                    ),
                    IC(
                        tags={'multi/to-langs', 'groupby/word'},
                        input='kobieta Frau -f pl de -t en es --groupby=word',
                        context={
                            'words': ['kobieta', 'Frau'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en', 'es'],
                        },
                        output='''
                            ──── en ──────────────────────────────────────
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ──── es ──────────────────────────────────────
                            kobieta: mujer (noun) [feminine], hembra (noun) [feminine], mujeres (noun) [feminine], femenino, señor, tía, mina, papel de la mujer, sexo femenino
                            Frau: mujer (noun) [feminine], esposa (noun) [feminine], señora (noun) [feminine], Señora, doña, marida, hembra, mujeres, Sra, fémina, individua, máquina, pariente, señor, dama, casar, cónyuge, persona, mina, galla, Sra., ella, sexo femenino, señorita, sra., tía, usted
                        ''',
                    ),
                    IC(
                        tags={'groupby/lang'},
                        input='kobieta Frau -f pl de -t en --groupby=lang',
                        context={
                            'words': ['kobieta', 'Frau'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en'],
                        },
                        output='''
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                        ''',
                    ),
                    IC(
                        tags={'groupby/word'},
                        input='kobieta Frau -f pl de -t en --groupby=word',
                        context={
                            'words': ['kobieta', 'Frau'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en'],
                        },
                        output='''
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                        ''',
                    ),
                    IC(
                        tags={'groupby/lang'},
                        input='kobieta Frau pan Herr -f pl de -t en --groupby=lang',
                        context={
                            'words': ['kobieta', 'Frau', 'pan', 'Herr'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en'],
                        },
                        output='''
                            ──── kobieta·Frau ────────────────────────────
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ──── pan·Herr ────────────────────────────────
                            pan: master (noun) [masculine], gentleman (noun) [masculine], sir (noun), mister, man, lord, Mr., lordship, Mr, Sir, Mister, gent, you, ruler, baboo, M., owner, Monsieur, between you and me, goodman, lording, monsieur, padrone, signore, the Lord, thou, thy, ye, yer, signor, squire, overlord, proprietor, swayer, possessor, adult male, pan, Senor, teacher, slaveholder, babu, buckra, instructor, man of means, men, mr., rich man, wealthy man
                            Herr: gentleman (noun) [masculine], Lord (proper) [noun], lord (noun) [masculine], sir, master, mister, man, Mr, Mister, gent, Mr., owner, Sir, you, signor, God, sovereign, male, esquire, shepherd, madam, god, boss, M., lordship, arbiter, baas, partner, ruler, sahib, the Lord, Father, chief, husband, host, leader, lover, mr., father, overlord, landlord, employer, guv, benefactor, saint, uncle, priest, mistress, almsgiver, bwana, slaveholder, almsgiving, alms, hostess, authority, Mrs., counter for people, gentleman (irr.), the main thing, the majority, the primary concern, this lady or gentleman
                        ''',
                    ),
                    IC(
                        tags={'groupby/word'},
                        input='kobieta Frau pan Herr -f pl de -t en --groupby=word',
                        context={
                            'words': ['kobieta', 'Frau', 'pan', 'Herr'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en'],
                        },
                        output='''
                            ━━━━ kobieta·Frau ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ━━━━ pan·Herr ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            pan: master (noun) [masculine], gentleman (noun) [masculine], sir (noun), mister, man, lord, Mr., lordship, Mr, Sir, Mister, gent, you, ruler, baboo, M., owner, Monsieur, between you and me, goodman, lording, monsieur, padrone, signore, the Lord, thou, thy, ye, yer, signor, squire, overlord, proprietor, swayer, possessor, adult male, pan, Senor, teacher, slaveholder, babu, buckra, instructor, man of means, men, mr., rich man, wealthy man
                            Herr: gentleman (noun) [masculine], Lord (proper) [noun], lord (noun) [masculine], sir, master, mister, man, Mr, Mister, gent, Mr., owner, Sir, you, signor, God, sovereign, male, esquire, shepherd, madam, god, boss, M., lordship, arbiter, baas, partner, ruler, sahib, the Lord, Father, chief, husband, host, leader, lover, mr., father, overlord, landlord, employer, guv, benefactor, saint, uncle, priest, mistress, almsgiver, bwana, slaveholder, almsgiving, alms, hostess, authority, Mrs., counter for people, gentleman (irr.), the main thing, the majority, the primary concern, this lady or gentleman
                        ''',
                    ),
                    IC(
                        tags={'groupby/lang'},
                        input='kobieta Frau pan Herr -f pl de -t en es --groupby=lang',
                        context={
                            'words': ['kobieta', 'Frau', 'pan', 'Herr'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en', 'es'],
                        },
                        output='''
                            ━━━━ en ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ──── kobieta·Frau ────────────────────────────
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ──── pan·Herr ────────────────────────────────
                            pan: master (noun) [masculine], gentleman (noun) [masculine], sir (noun), mister, man, lord, Mr., lordship, Mr, Sir, Mister, gent, you, ruler, baboo, M., owner, Monsieur, between you and me, goodman, lording, monsieur, padrone, signore, the Lord, thou, thy, ye, yer, signor, squire, overlord, proprietor, swayer, possessor, adult male, pan, Senor, teacher, slaveholder, babu, buckra, instructor, man of means, men, mr., rich man, wealthy man
                            Herr: gentleman (noun) [masculine], Lord (proper) [noun], lord (noun) [masculine], sir, master, mister, man, Mr, Mister, gent, Mr., owner, Sir, you, signor, God, sovereign, male, esquire, shepherd, madam, god, boss, M., lordship, arbiter, baas, partner, ruler, sahib, the Lord, Father, chief, husband, host, leader, lover, mr., father, overlord, landlord, employer, guv, benefactor, saint, uncle, priest, mistress, almsgiver, bwana, slaveholder, almsgiving, alms, hostess, authority, Mrs., counter for people, gentleman (irr.), the main thing, the majority, the primary concern, this lady or gentleman
                            ━━━━ es ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ──── kobieta·Frau ────────────────────────────
                            kobieta: mujer (noun) [feminine], hembra (noun) [feminine], mujeres (noun) [feminine], femenino, señor, tía, mina, papel de la mujer, sexo femenino
                            Frau: mujer (noun) [feminine], esposa (noun) [feminine], señora (noun) [feminine], Señora, doña, marida, hembra, mujeres, Sra, fémina, individua, máquina, pariente, señor, dama, casar, cónyuge, persona, mina, galla, Sra., ella, sexo femenino, señorita, sra., tía, usted
                            ──── pan·Herr ────────────────────────────────
                            pan: señor (noun) [masculine], amo (noun) [masculine], caballero (noun) [masculine], hombre, varón, don, maese, Señor, usted, señoría, vosotros, dueño, gobernador, D., cacerola, sr., ustedes, se, te, ti, pan, patrón, propietario, poseedor, dueco, posesor, caballeros, el Señor, hombres, vos
                            Herr: señor (noun) [masculine], Señor (proper) [masculine], caballero (noun) [masculine], amo, hombre, varón, señores, dueño, Dios, don, usted, Sr, jefe, patrón, soberano, padre, Don, Sr., santo, comandante, general, grande, patrono, líder, anciano, papa, capitán, tío, papá, caudillo, brío, pieza, autoridad
                        ''',
                    ),
                    IC(
                        tags={'groupby/word'},
                        input='kobieta Frau pan Herr -f pl de -t en es --groupby=word',
                        context={
                            'words': ['kobieta', 'Frau', 'pan', 'Herr'],
                            'from_langs': ['pl', 'de'],
                            'to_langs': ['en', 'es'],
                        },
                        output='''
                            ━━━━ kobieta·Frau ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ──── en ──────────────────────────────────────
                            kobieta: woman (noun), female (noun), dame (noun), lady, missus, femme, women, she, skirt, old lady, wife, bit of skirt, broad, gal, bitch, mama, fem, baggage, petticoat, cummer, feme, missis, adult female, role of women, status of women
                            Frau: woman (noun), wife (noun), lady (noun), Mrs, madam, spouse, mistress, female, Ms, girl, Mrs., Lady, women, femme, dame, mrs, signora, Frau, womankind, embellishment, husband, missus, adult female, consort of the emperor, feminine gender, grown woman, woman writer, Goodwife, bride, judy, feminine, mate, broad, daughter, Woman, jane, Female, consort, lover, she, sheila, Miss, eldest daughter, female spouse, huge breasts, womenfolk, excellent, missis, inside, superb, chignon, interior, Mr., Ms., inner part, married woman, wicked woman, woman (irr.), your wife
                            ──── es ──────────────────────────────────────
                            kobieta: mujer (noun) [feminine], hembra (noun) [feminine], mujeres (noun) [feminine], femenino, señor, tía, mina, papel de la mujer, sexo femenino
                            Frau: mujer (noun) [feminine], esposa (noun) [feminine], señora (noun) [feminine], Señora, doña, marida, hembra, mujeres, Sra, fémina, individua, máquina, pariente, señor, dama, casar, cónyuge, persona, mina, galla, Sra., ella, sexo femenino, señorita, sra., tía, usted
                            ━━━━ pan·Herr ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ──── en ──────────────────────────────────────
                            pan: master (noun) [masculine], gentleman (noun) [masculine], sir (noun), mister, man, lord, Mr., lordship, Mr, Sir, Mister, gent, you, ruler, baboo, M., owner, Monsieur, between you and me, goodman, lording, monsieur, padrone, signore, the Lord, thou, thy, ye, yer, signor, squire, overlord, proprietor, swayer, possessor, adult male, pan, Senor, teacher, slaveholder, babu, buckra, instructor, man of means, men, mr., rich man, wealthy man
                            Herr: gentleman (noun) [masculine], Lord (proper) [noun], lord (noun) [masculine], sir, master, mister, man, Mr, Mister, gent, Mr., owner, Sir, you, signor, God, sovereign, male, esquire, shepherd, madam, god, boss, M., lordship, arbiter, baas, partner, ruler, sahib, the Lord, Father, chief, husband, host, leader, lover, mr., father, overlord, landlord, employer, guv, benefactor, saint, uncle, priest, mistress, almsgiver, bwana, slaveholder, almsgiving, alms, hostess, authority, Mrs., counter for people, gentleman (irr.), the main thing, the majority, the primary concern, this lady or gentleman
                            ──── es ──────────────────────────────────────
                            pan: señor (noun) [masculine], amo (noun) [masculine], caballero (noun) [masculine], hombre, varón, don, maese, Señor, usted, señoría, vosotros, dueño, gobernador, D., cacerola, sr., ustedes, se, te, ti, pan, patrón, propietario, poseedor, dueco, posesor, caballeros, el Señor, hombres, vos
                            Herr: señor (noun) [masculine], Señor (proper) [masculine], caballero (noun) [masculine], amo, hombre, varón, señores, dueño, Dios, don, usted, Sr, jefe, patrón, soberano, padre, Don, Sr., santo, comandante, general, grande, patrono, líder, anciano, papa, capitán, tío, papá, caudillo, brío, pieza, autoridad
                        ''',
                    ),
                ],
            ),
            TC(
                descr='Sidality',
                tags={'side', 'at'},
                input=[
                    IC(
                        tags={'nontranslate', 'at', 'wiktio', 'overview', 'definition', 'inflection'},
                        input={f'Frau de -{"".join(perm)}' for perm in permutations('doi', 3)},
                        context={
                            'at': 'none',
                            'definition': True,
                            'inflection': True,
                            'wiktio': True,
                            'words': {'Frau'},
                            'from_langs': ('de',),
                        },
                        output=''.join((frau_inflection.rstrip(), frau_overview.rstrip(), frau_definition)),
                    ),
                    IC(
                        tags={'translate', 'side/from', 'at', 'wiktio', 'overview', 'definition', 'inflection'},
                        input={f'Frau de -f{"".join(perm)}' for perm in permutations('doi', 3)},
                        context={
                            'at': 'f',
                            'definition': True,
                            'inflection': True,
                            'wiktio': True,
                            'words': {'Frau'},
                            'from_langs': ('de',),
                        },
                        output=''.join((
                            frau_inflection.rstrip(),
                            frau_translation.rstrip(),
                            '\n'.join(line for line in frau_overview.split('\n') if 'Frau:' not in line),
                            frau_definition.replace(' of "Frau"', '')
                        )),
                    ),
                    IC(
                        tags={'translate', 'side/to', 'at', 'wiktio', 'overview', 'definition', 'inflection'},
                        input={f'Frau de -t{"".join(perm)}' for perm in permutations('doi', 3)},
                        context={
                            'at': 't',
                            'definition': True,
                            'inflection': True,
                            'wiktio': True,
                            'words': {'Frau'},
                            'from_langs': ('de',),
                        },
                        output='''
                            ╭───┬──────────────┬──────────┬───────────╮
                            │ 0 │ nominative   │ kobieta  │ kobiety   │
                            │ 1 │ genitive     │ kobiety  │ kobiet    │
                            │ 2 │ dative       │ kobiecie │ kobietom  │
                            │ 3 │ accusative   │ kobietę  │ kobiety   │
                            │ 4 │ instrumental │ kobietą  │ kobietami │
                            │ 5 │ locative     │ kobiecie │ kobietach │
                            │ 6 │ vocative     │ kobieto  │ kobiety   │
                            ╰───┴──────────────┴──────────┴───────────╯
                            Frau: kobieta (noun) [feminine], żona (noun) [feminine], pani (noun) [abbreviation], małżonka, babka, mężatka, lady, Pani, kobièta, dama, WPani, facetka, kobita, kobitka, panna, ona, niewiasta, baba, samica, białogłowa, dupa, babsko, czyściocha, jejmość, p.
                            meanings:
                              • /kɔˈbjɛ.ta/, /kɔˈbje.ta/ [PoS: Noun, gender: f, diminutive: kobietka, augmentative: kobiecisko, related adjective: kobiecy or (obsolete) kobiecki]
                                etymology:
                                  - Uncertain.[1] Displaced niewiasta (now considered poetic) and żona (meaning narrowed down to “wife”). Popularized by the Polish Romantic writers in the 19th century. First attested in 1545.[2]
                            meanings:
                              • /kɔˈbjɛ.ta/, /kɔˈbje.ta/ [PoS: Noun, gender: f, diminutive: kobietka, augmentative: kobiecisko, related adjective: kobiecy or (obsolete) kobiecki]
                                etymology:
                                  - Uncertain.[1] Displaced niewiasta (now considered poetic) and żona (meaning narrowed down to “wife”). Popularized by the Polish Romantic writers in the 19th century. First attested in 1545.[2]
                            
                            Definitions:
                            - Weib (derb)
                            - Weibsstück (abwertend) (derb)
                            - Eine verheiratete Frau.
                            - Anrede für einen erwachsenen, weiblichen Menschen.
                            - engl. Anrede, die keinen Unterschied zwischen verheiratet und unverheiratet macht
                            - Slang
                            - unhöflich
                            - höfl. für Frau oder Kinder
                            - Erwachsene, menschliche Angehörige des Geschlechts, das Eizellen produziert und Kinder gebärt.

                        ''',
                    ),
                ],
                conf=assume_langs_pl_de_en_es_ru,
            ),
            TC(
                descr='Mapping',
                tags={'mapping'},
                input=[
                    IC(
                        tags={'mapping/simple'},
                        input='eo sxangxi pl',
                        context={'words': ['ŝanĝi']},
                    ),
                    IC(
                        tags={'mapping/sequential'},
                        input='uk żurawel pl',
                        context={'words': ['журавель']},
                    ),
                    IC(
                        tags={'mapping/sequential'},
                        input={'uk pl ziłłia', 'uk pl ziłla', 'uk pl ziłlia'},
                        context={'words': ['зілля']},
                    ),
                    IC(
                        tags={'mapping/sequential'},
                        input={'uk pl miljard', 'uk pl "mił\'iard"', 'uk pl "mil\'iard"'},
                        context={'words': ['мільярд']}
                    ),
                    IC(
                        tags={'mapping/multi-lang'},
                        input='-f uk eo -w zminyty sxangxi -t pl',
                        context={'words': ['змінити', 'ŝanĝi']}
                    )
                ],
                skip_mocking=True,
                conf=Box({
                    'langs': ['pl', 'eo', 'uk'],
                    'mappings': {
                        'eo': {'Cx': 'Ĉ', 'Gx': 'Ĝ', 'Hx': 'Ĥ', 'Jx': 'Ĵ', 'Sx': 'Ŝ', 'Ux': 'Ŭ', 'cx': 'ĉ', 'gx': 'ĝ', 'hx': 'ĥ', 'jx': 'ĵ', 'sx': 'ŝ', 'ux': 'ŭ'},
                        'uk': [
                            {'([aeouy])i': '\\1ji', 'l([aeuo])': 'łi\\1', 'li': 'łi'},
                            {'[ji]a': 'я', '[ji]e': 'є', '[ji]u': 'ю', 'ji': 'ї'},
                            {"[q']": 'ь'},
                            {'["x]': "'", 'a': 'а', 'b': 'б', 'c': 'ц', 'ch': 'х', 'cz': 'ч', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'ґ', 'h': 'г', 'i': 'і', 'j': 'й', 'k': 'к', 'l': 'ль', 'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'r': 'р', 's': 'с', 'sz': 'ш', 'szcz': 'щ', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': "'", 'y': 'и', 'z': 'з', 'ć': 'ць', 'ł': 'л', 'ń': 'нь', 'ś': 'сь', 'ż': 'ж'},
                            {'ьь': 'ь'},
                        ]
                    }
                }),
            )
    ]

    # TODO: fix or inform: 标 zh -o

    @classmethod
    def map_to_many(cls, tc: TC) -> Iterable[Tc]:
        tcs = cls.map_inputs(tc)  # input > InputCase (single)
        tcs = _.flat_map(tcs, cls.replace_placeholders)
        tcs = _.map_(tcs, cls.singularize)
        return tcs

    @classmethod
    def map_inputs(cls, tc: TC) -> list:
        match tc.input:
            case str(): return [replace(tc, input=InputCase(input=tc.input))]
            case _ as i if isinstance(i, Iterable):
                return _.flat_map((replace(tc, input=input) for input in tc.input), cls.map_inputs)
            case InputCase():
                match tc.input.input:
                    case str(): return [tc]
                    case _ as ii if isinstance(ii, Iterable):
                        return [replace(tc, input=replace(tc.input, input=input)) for input in tc.input.input]
                    case _ as ii: raise ValueError(f'Unexpected input type: {type(ii)}')
            case _: raise ValueError(f'Unexpected input type: {type(tc.input)}')

    @classmethod
    def replace_placeholders(cls, tc: TC) -> Iterable[TC]:
        def replace_at(replacements: dict[str, list], tc: TC):
            for placetaker_batch in product(*replacements.values()):
                curr_tc = replace(tc)
                curr_input: str = tc.input.input
                for placeholder, placetaker in zip(replacements.keys(), placetaker_batch):
                    old_input = str(curr_input)
                    curr_input = curr_input.replace(f'<{placeholder}>', placetaker)
                    if curr_input != old_input:
                        curr_tc = replace(curr_tc, tags=list(curr_tc.tags or []) + [f'replacement/{placetaker}'])
                curr_tc = replace(curr_tc, input=replace(tc.input, input=curr_input))
                yield curr_tc
        tcs = replace_at(tc.input.replacement, tc)
        tcs = _.uniq_by(tcs, lambda tc: tc.input.input)
        top_repl_tcs = (subtc for tc in tcs for subtc in replace_at(tc.replacement, tc))
        tcs = c(top_repl_tcs).uniq_by(lambda tc: tc.input.input).value()
        return tcs

    @classmethod
    def singularize(cls, tc: TC) -> Tc:
        single = Tc(
            descr=tc.descr,
            tags=list(tc.tags) + list(tc.input.tags),
            input=tc.input.input,
            context=Box({key: cls.map_context_val(val) for key, val in {**tc.context, **tc.input.context}.items()}).to_dict(),
            output=cls.regularize_output(tc.input.output or tc.output),
            exception=c([tc.input.exception] + [tc.exception]).flatten().filter().value(),
            conf=Box({**tc.conf, **tc.input.conf}).to_dict(),
            color=tc.color,
            skip_mocking=tc.skip_mocking or tc.input.skip_mocking,
        )
        if '-h' in single.input:
            single.exception.append(SystemExit)
        return single

    @classmethod
    def map_context_val(cls, val: Any) -> Any:
        match val:
            case str(): return val
            case _ if isinstance(val, (Collection, Iterable)) and not isinstance(val, str):
                return list(val)
            case _: return val

    @classmethod
    def regularize_output(cls, output: str) -> str:
        if not output or len(lines := output.split('\n')) < 3:
            return output
        lines = lines[1:-1]
        n_spaces = len(lines[0]) - len(lines[0].lstrip(' '))
        refined = c(lines).map(lambda line: line[n_spaces:].rstrip()).join('\n').value()
        return refined

    @classmethod
    def gather_tags(cls, tc: Tc) -> Iterable[str]:
        tc.tags += [f'conf/{key}/{val}' for key, val in tc.conf.items()]
        tc.tags += [f'flag/{flag}' for flag in shlex.split(tc.input) if flag.startswith('-')]
        tc.tags += [f'context/should/{key}/{list(val) if isinstance(val, list) else val}' for key, val in tc.context.items()]
        if '--set' in tc.input:
            tc.tags += [f'conf/set/{key.replace("_conf.")}/{val}' for key, val in tc.context.items() if '_conf.' in key]
        return tc.tags

    @classmethod
    def create_name(cls, tc) -> Optional[str]:
        return f'{tc.descr}: "t {tc.input}", ({tc.tags})'

@pytest.fixture(autouse=True)
def patch(mocker):
    mocker.patch(
        'src.scrapping.core.scrap_adapting.ScrapAdapter.scrap',
        side_effect=mocked_scrap,
    )
    mocker.patch(
        'src.app_managing.AppMgr.connect',
        return_value=nullcontext(None),
    )


@SystemTCG.parametrize('tc')
def test(tc: Tc):
    with open(TEST_CONF, 'w') as f:
        yaml.dump(tc.conf, f, default_flow_style=False, allow_unicode=True)
    collector = CallCollector(line_mapper=c().trim_end())
    app_mgr = AppMgr(conf_path=TEST_CONF, printer=collector)

    exceptions = []
    if tc.exception:
        exceptions.extend(tuple(tc.exception))
    if tc.skip_mocking:
        exceptions.append(PageNotFound)
    ctx = pytest.raises(tuple(exceptions)) if exceptions else nullcontext()
    with ctx:
        app_mgr.run_single(shlex.split(tc.input))

    for path, e_val in tc.context.items():
        a_val = _.get(app_mgr.context, path)
        if isinstance(e_val, (Collection, Iterable)) and not isinstance(e_val, str) and isinstance(a_val, (Collection, Iterable)):
            a_val = list(a_val)
        assert path == path and e_val == a_val

    if not tc.skip_mocking and tc.output:
        a_output = (_.identity if tc.color else remove_color)(collector.output)
        assert tc.output == a_output
