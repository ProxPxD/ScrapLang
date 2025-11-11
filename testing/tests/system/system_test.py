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
from testing.core import TCG
from testing.core.mocking import mocked_scrap, CallCollector

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

    replacement: dict = field(default_factory=dict)

IC = InputCase

@dataclass
class TC:  # TODO: think of disabling queries
    descr: str
    input: Iterable[InputCase | str] | InputCase | str
    tags: Iterable[str] = field(default_factory=list)
    conf: dict[str, Any] = field(default_factory=Box)
    context: dict[str, Any] = field(default_factory=dict)
    output: str = ''

    replacement: dict = field(default_factory=dict)



@dataclass
class Tc:
    descr: str
    tags: Sequence[str]

    input: str
    conf: dict
    context: dict
    output: str


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
        'from_lang': 'en',
        'to_langs': ('pl', 'de'),
        'words': ('water', 'bass'),
    })
    multi_perm = Box({
        'langs': (perm_base.from_lang, *perm_base.to_langs),
        'args': (perm_base.from_lang, *perm_base.to_langs, *perm_base.words),
        **perm_base,
    })

    @classmethod
    def generate_tcs(cls) -> list:
        return [
        TC(
            descr='Exact allflag configless call',
            tags=['position', 'permutation'],
            input=_.map_(permutations({'-w Herr', '-f de', '-t pl'}, 3), ' '.join),
            context={
                'from_lang': 'de',
                'words': frozenset({'Herr'}),
                'to_langs': frozenset({'pl'}),
            },
            output='Herr: pan (noun) [masculine], mężczyzna (noun) [masculine], Pan (proper) [masculine], człowiek, dżentelmen, jegomość, mąż, panowie, mistrz, don, partner, Bóg, władca, szef, gospodarz, zwierzchnik, nauczyciel, majster, p., panie'
        ),
        TC(
            descr=f'Conf Loading',
            tags=['conf', 'load'],
            input=[
                IC(
                    input=f'-h',  # TODO think
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
            input={
                'es de -w en de orden --assume lang',  # lang
                'en de orden --assume word',  # word
                'es de -w en de orden',  # Default
            },
            context={
                'from_lang': ('es'),
                'words': frozenset({'en', 'de', 'orden'}),
                'to_langs': frozenset({'de'}),
            },
            conf=(just_langs_es_de_pl_en_conf := Box({
                'langs': ['es', 'de', 'pl', 'en'],
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
            context={
                'from_lang': ('es'),
                'to_langs': frozenset({'pl', 'de'})  # TODO: Error no word
            },
            conf=just_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Single translation of any arg placement resolution',
            tags={'permutation', 'position', 'single-from-lang', 'single-to-lang', 'single-word'},
            input={
                'Herr de pl',
                'de Herr pl',
                'de pl Herr',
            },
            context={
                'from_lang': 'de',
                'words': frozenset({'Herr'}),
                'to_langs': frozenset({'pl'}),
            },
            conf=(base_langs_es_de_pl_en_conf := Box(dict(**just_langs_es_de_pl_en_conf, assume='lang'))),
        ),
        TC(
            descr='Multiword from single lang to multilang translation of any arg placement resolution',
            tags={'permutation', 'position', 'multi-word', 'multi-to-lang', 'single-from-lang'},
            input={
                ' '.join(perm)
                for perm
                in permutations(cls.multi_perm.args, len(cls.multi_perm.args))
                if all(i < j for group in (cls.multi_perm.langs, cls.multi_perm.words) for i, j in windowed(map(perm.index, group), 2))
            },
            context={
                'from_lang': ('en'),
                'words': frozenset({'water', 'bass'}),
                'to_langs': frozenset({'pl', 'de'}),
            },
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Main Replacements',
            tags=set(),
            input={
                'es pl orden conocer <WORD_FLAG> en de',
                'pl orden conocer -w en de <FROM_LANG> es',
                'es orden conocer -w en de <TO_LANG> pl',
            },
            replacement={
                'WORD_FLAG': ['--word', '--words', '-word', '-words'],
                'FROM_LANG': ['--from-lang', '--from', '-from'],
                'TO_LANG':   ['--to-lang', '--to-langs', '--to', '-to'],
            },
            context={
                'words': frozenset({'orden', 'conocer', 'en', 'de'}),
                'from_lang': ('es'),
                'to_langs': frozenset({'pl'}),
            },
            conf=base_langs_es_de_pl_en_conf,
        ),
        TC(
            descr='Inflection',
            tags=set(),
            input=[
                IC(
                    tags={'permutation', 'position'},
                    input=_.map_(permutations({'Herr', 'de', '-i'}, 3), ' '.join),
                    context=(herr_context := {
                        'from_lang': ('de'),
                        'to_langs': frozenset(),
                        'inflection': True,
                        'words': ['Herr']
                    }),
                    output='''
                        ╭───┬────────────┬───────┬─────┬───────┬─────┬────────╮
                        │ 0 │ nominative │ ein   │ der │ Herr  │ die │ Herren │
                        │ 1 │ genitive   │ eines │ des │ Herrn │ der │ Herren │
                        │ 2 │ dative     │ einem │ dem │ Herrn │ den │ Herren │
                        │ 3 │ accusative │ einen │ den │ Herrn │ die │ Herren │
                        ╰───┴────────────┴───────┴─────┴───────┴─────┴────────╯
                    '''
                ),
                IC(
                    input='Herr de <INFL_FLAG>',
                    replacement={'INFL_FLAG': ['--inflection', '--infl', '-infl']},
                    context=herr_context,
                ),
                # TODO: extend with concrete inflection types and tables to test
            ],
            conf=(assume_langs_pl_de_en_es := Box({
                'langs': ['pl', 'de', 'en', 'es'],
            })),
        ),
        TC(
            descr='Definition',
            tags=set(),
            input=[
                IC(
                    tags={'permutation', 'position'},
                    input=_.map_(permutations({'bass', 'en', '-d'}, 3), ' '.join),
                    context=(bass_context := {
                        'from_lang': ('en'),
                        'definition': True,
                        'words': ['bass']
                    }),
                    output='''
                        Definitions of "bass":                                                                                                                                                                                                            
                        - Of sound, a voice or an instrument, low in pitch or frequency.                                                                                                                                                                  
                        - A low spectrum of sound tones.                                                                                                                                                                                                  
                        - A section of musical group that produces low-pitched sound, lower than tenor.                                                                                                                                                   
                        - A male singer who sings in the bass range.
                        - An instrument that plays in the bass range, in particular a double bass, bass guitar, electric bass or bass synthesiser.
                        - The clef sign that indicates that the pitch of the notes is below middle C; a bass clef.
                        - The perch; any of various marine and freshwater fish resembling the perch, all within the order of Perciformes.
                        - (adjective) Of sound, a voice or an instrument, low in pitch or frequency.
                        - (noun) The perch; any of various marine and freshwater fish resembling the perch, all within the order of Perciformes.
                        - (noun) The fibrous inner bark of the linden or lime tree, used for making mats.
                        - (noun) Fibers from other plants, especially palm trees
                        - (noun) Anything made from such fibers, such as a hassock or thick mat.
                        - (noun) A low spectrum of sound tones.
                        - (noun) A section of musical group that produces low-pitched sound, lower than the baritone and tenor.
                        - (noun) A male singer who sings in the bass range.
                        - (noun (music)) An instrument that plays in the bass range, in particular a double bass, bass guitar, electric bass or bass synthesiser.
                        - (noun) The clef sign that indicates that the pitch of the notes is below middle C; a bass clef.
                        - (verb) To sound in a deep tone.
                        - A marine fish (Percicthyidae or Centrarchidae) that is popular as game.
                        - A musical part or section lower than tenor and all other sections, with a typical range from the D below the bottom of the bass clef to the E above "middle C".
                        - low in pitch
                        - low spectrum of sound
                        - section of musical group
                        - singer
                        - musical instrument
                        - clef sign
                        - perch
                        - A male singer who sings in the deepest vocal range.
                        - of low frequency or range.
                        - A sound range of low frequency or range.
                        - A musical clef indicating that the F3 note is placed on the fourth line.
                        - An instrument that plays sounds of low frequency.
                        - (adjective) having or denoting a low vocal or instrumental range; "a deep voice"; "a bass voice is lower than a baritone voice"; "a bass clarinet"
                        - (noun) any of various North American freshwater fish with lean flesh (especially of the genus Micropterus)
                        - (noun) an adult male singer with the lowest voice
                        - (noun) nontechnical name for any of numerous edible marine and freshwater spiny-finned fishes
                        - (noun) the lean flesh of a saltwater fish of the family Serranidae
                        - (noun) the lowest adult male singing voice
                        - (noun) the lowest part in polyphonic music
                        - (noun) the lowest part of the musical range
                        - (noun) the member with the lowest range of a family of musical instruments
                        - (proper) A surname.
                        - One of the music genres that appears under Genre classification in Windows Media Player library. Based on ID3 standard tagging format for MP3 audio files. ID3v1 genre ID # 41.
                        - One of the music genres that appears under Genre classification in Windows Media Player library. Based on ID3 standard tagging format for MP3 audio files. Winamp genre ID # 127.
                    
                    ''',
                ),
                IC(
                    input='bass <DEF_FLAG> en',
                    replacement={
                        'DEF_FLAG': ['--definitions', '--definition', '-definitions', '-definition', '--def', '-def']},
                    context=bass_context,
                ),
                # TODO: extend with definition formatting
            ],
            conf=assume_langs_pl_de_en_es,
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
                        'from_lang': ('es'),
                        'to_langs': frozenset({'pl'}),
                        'words': ['conocer']
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
                        -------- pl -------------------------
                        conocer: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                        orden: porządek (noun) [masculine], rząd (noun) [masculine], rozkaz (noun) [masculine], zakon, zamówienie, polecenie, nakaz, kolejność, order, zarząd, zarządzenie, instrukcja, dyspozycja, kategoria, komenda, odznaczenie, rozporządzenie, szyk, układ, zalecenie, zamówić, wyrok, ustawa, zamawiać, przykazanie, zlecenie, relacja porządkowa, relacja porządku, ład, wskazówka, ustawodawstwo, prawodawstwo, konta, odpisu, uporządkowanie
                        -------- de -------------------------
                        conocer: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
                        orden: Ordnung (noun) [masculine], Befehl (noun) [feminine], Reihenfolge (noun) [masculine], Orden, Kommando, Auftrag, Anordnung, Anweisung, Gebot, Bestellung, ordnung, Order, Geheiß, Verfügung, Regierung, Behörde, Leitung, Regieren, befehlen, Reihe, Vorstand, verordnung, Vorschrift, bestellen, serie, Bereich, Erlass, Geschäftsordnung, Instruktion, Komplex, Königswürde, Ordentlichkeit, Rang, Rangordnung, Recht, Systematik, Säulenordnung, Veranlassung, Weihe, Weihen, Ordnungsrelation, Aneinanderreihung, Abbuchungsauftrag, Verkettung, Organisation, Richtlinie, Sinn, zeitliche Ordnung, Verordnung, Weisung, Dienst, Gesetz, System, Sortierung, Diktat, Klasse, Ordo, Serie, Kategorie, Gesetzesentwurf, Gesetzgebung, Gerichtsurteil, Methode, Orden und Ehrenzeichen, Urteil, orden und ehrenzeichen#verdienstauszeichnungen
                    '''
                ),
                IC(
                    tags={'flag/groupby/word'},
                    input='es pl de -w conocer orden -by word',
                    output='''
                        -------- conocer -------------------------
                        pl: znać (Modal) [impf], znany (adjective), poznać (verb), poznawać, poznać się, znać się, wiedzieć, umieć, wiedzieć z góry
                        de: kennen (verb) [v], kennenlernen (verb), verstehen (verb), bekannt, lernen, begreifen, befinden, sehen, sich auskennen, urteilen, wissen, erkennen, auskennen, besichtigen, kennen lernen
                        -------- orden -------------------------
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
    ]

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
        return Tc(
            descr=tc.descr,
            tags=list(tc.tags) + list(tc.input.tags),
            input=tc.input.input,
            context=Box(
                {key: cls.map_context_val(val) for key, val in {**tc.context, **tc.input.context}.items()}
            ).to_dict(),
            output=cls.regularize_output(tc.input.output or tc.output),
            conf=Box({**tc.conf, **tc.input.conf}).to_dict(),
        )

    @classmethod
    def map_context_val(cls, val: Any) -> Any:
        match val:
            case str(): return val
            case _ if isinstance(val, (Collection, Iterable)) and not isinstance(val, str):
                return frozenset(val)
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
        tc.tags += [f'context/should/{key}/{list(val) if isinstance(val, frozenset) else val}' for key, val in tc.context.items()]
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
    app_mgr = AppMgr(conf_path=TEST_CONF, printer=(collector := CallCollector()))

    ctx = pytest.raises(SystemExit) if '-h' in tc.input else nullcontext()
    with ctx:
        app_mgr.run_single(shlex.split(tc.input))

    for path, e_val in tc.context.items():
        a_val = _.get(app_mgr.context, path)
        if isinstance(e_val, (Collection, Iterable)) and not isinstance(e_val, str):
            a_val = frozenset(a_val)
        assert path == path and e_val == a_val

    if tc.output:
        assert tc.output == collector.output