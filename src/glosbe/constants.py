from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FLAGS:
    @dataclass(frozen=True)
    class MODES:
        SINGLE_LONG_FLAG = '--single'
        LANG_LONG_FLAG = '--lang'
        WORD_LONG_FLAG = '--word'
        SINGLE_SHORT_FLAG = '-s'
        LANG_SHORT_FLAG = '-m'
        WORD_SHORT_FLAG = '-w'

    @dataclass(frozen=True)
    class CONFIGURATIONAL:
        LANG_LIMIT_LONG_FLAG = '--limit'
        LANG_LIMIT_SHORT_FLAG = '-l'
        LANGS_SHOW_LONG_FLAG = '--langs'
        LANGS_SHOW_SHORT_FLAG = '-ll'
        LAST_LANG_LONG_FLAG = '--last'  # put number
        LAST_1_LONG_FLAG = '--last1'
        LAST_1_SHORT_FLAG = '-1'
        LAST_2_LONG_FLAG = '--last2'
        LAST_2_SHORT_FLAG = '-2'
        DEFAULT_MODE_LONG_FLAG = '--default-mode'
        DEFAULT_MODE_SHORT_FLAG = '-dm'
        SETTINGS_LONG_FLAG = '--settings'
        SETTINGS_SHORT_FLAG = '-ss'
        DOUBLE_MODE_STYLE_LONG_FLAG = '--double-mode-style'  # poss: lang word single/double
        DOUBLE_MODE_STYLE_SHORT_FLAG = '-ds'
        LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG = '--layout-adjustment-method'
        LAYOUT_ADJUSTMENT_METHOD_SHORT_FLAG = '-lam'
        LAYOUT_ADJUSTMENT_LANG_LONG_FLAG = '--layout-adjustment-lang'
        LAYOUT_ADJUSTMENT_LANG_SHORT_FLAG = '-lal'
        ADD_LANG_LONG_FLAG = '--add-lang'
        ADD_LANG_SHORT_FLAG = '-al'
        REMOVE_LANG_LONG_FLAG = '--remove-lang'
        REMOVE_LANG_SHORT_FLAG = '-rl'
        SHOW_INFO_MODE_FLAG_LONG = '--show-info-mode'
        SHOW_INFO_MODE_FLAG_SHORT = '-si'

    @dataclass(frozen=True)
    class FUNCTIONAL:
        SILENT_LONG_FLAG = '--silent'
        REVERSE_LONG_FLAG = '--reverse'
        REVERSE_SHORT_FLAG = '-r'
        SYNOPSIS_LONG_FLAG = '--synopsis'
        SYNOPSIS_SHORT_FLAG = '-syn'
        FROM_LANG_LONG_FLAG = '--from'
        FROM_LANG_SHORT_FLAG = '-f'
        CONJUGATION_LONG_FLAG = '--conjugation'
        CONJUGATION_SHORT_FLAG = '-conj'
        CONJUGATION_SUPER_SHORT_FLAG = '-c'
        CCONJUGATION_LONG_FLAG = '--cconjugation'
        CCONJUGATION_SHORT_FLAG = '-cconj'
        CCONJUGATION_SUPER_SHORT_FLAG = '-cc'
        DEFINITION_LONG_FLAG = '--definition'
        DEFINITION_MID_FLAG = '-def'
        DEFINITION_SHORT_FLAG = '-d'
        INFO_LONG_FLAG = '--info'
        INFO_SHORT_FLAG = '-i'

    HELP_LONG_FLAG = '--help'
    HELP_SHORT_FLAG = '-h'

    C = CONFIGURATIONAL
    F = FUNCTIONAL
    M = MODES


@dataclass(frozen=True)
class Paths:
    WORKING_DIR = Path(__file__).parent.parent.parent
    RESOURCES_DIR = WORKING_DIR / 'resources'
    LOG_DIR = RESOURCES_DIR / 'logs.log'
    CONF_FILE: str = RESOURCES_DIR / 'conf.yaml'


supported_languages = {
    'af': "afrikaans Afrikaans",
    'sq': "albański Albanian, shqip",
    'en': "angielski English",
    'ar': "arabski Arabic, العربية",
    'az': "azerski Azerbaijani, azərbaycan dili",
    'eu': "baskijski Basque, Euskara, Euskera, Vascuense, euskara",
    'bn': "bengalski Bangla, Bangala, Bangla-Bhasa, বাংলা",
    'be': "białoruski Belarusian, Belorussian, Bielorussian, Byelorussian, White Russian, White Ruthenian, беларуская",
    'my': "birmański Burmese, Bama, Bamachaka, Myanmar, Myen, ဗမာ",
    'bs': "bośniacki Bosnian, bosanski",
    'bg': "bułgarski Bulgarian, Balgarski, български",
    'zh': "chiński Chinese, 中文",
    'hr': "chorwacki Croatian, Hrvatski, hrvatski",
    'cs': "czeski Czech, Bohemian, Cestina, čeština",
    'nds': "dolnosaksoński Low German, Nedderdütsch, Neddersassisch, Nedersaksisch, Niedersaechsisch, Plattdeutsch, Plattdüütsch, Platduuts",
    'da': "duński Danish, Dansk, Rigsdansk, dansk",
    'eo': "esperanto Eo, La Lingvo Internacia, Esperanto",
    'et': "estoński Estonian, eesti",
    'fil': "filipino Filipino",
    'fi': "fiński Finnish, Suomi, suomi",
    'fr': "francuski French, Français, français",
    'gl': "galisyjski Galician, Galego, Gallego, galego",
    'el': "grecki Greek, Ellinika, Graecae, Grec, Greco, Neo-Hellenic, Romaic, Ελληνικά",
    'ka': "gruziński Georgian, Common Kartvelian, Gruzinski, Kartuli, ქართული",
    'gu': "gudźaracki Gujarati, Gujerathi, Gujerati, Gujrathi, ગુજરાતી",
    'ht': "haitański Haitian, Creole, Haitian Creole, Kreyòl ayisyen",
    'he': "hebrajski Hebrew, Israeli, Ivrit, עברית",
    'hi': "hindi Hindi, Khadi Boli, Khari Boli, हिन्दी",
    'es': "hiszpański Spanish, Castellano, Castilian, Español, español",
    'io': "ido Ido",
    'id': "indonezyjski Indonesian, Bahasa Indonesia, Indonesia",
    'ia': "interlingua Interlingua",
    'ga': "irlandzki Irish, Erse, Gaelic Irish, Gaeilge",
    'ja': "japoński Japanese, 日本語",
    'yo': "joruba Yoruba, Yariba, Yooba, Èdè Yorùbá",
    'ca': "kataloński Catalan, Català, Catalan-Valencian-Balear, Catalonian, català",
    'kk': "kazachski Kazakh, Kaisak, Kazak, Kosach, Qazaq, Қазақ тілі",
    'km': "khmerski Khmer, Cambodian, ខ្មែរ",
    'ko': "koreański Korean, Hanguk Mal, Hanguk Uh, 한국어",
    'ku': "kurdyjski Kurdish Kurmanji, Zimanê kurdî",
    'lo': "laotański Lao, Eastern Thai, Lào, Lao Kao, Lao Wiang, Lao-Lum, Lao-Noi, Lao-Tai, Laotian, Laotian Tai, Lum Lao, Phou Lao, Rong Kong, Tai Lao, ລາວ",
    'lt': "litewski Lithuanian, Lietuvi, Lietuviskai, Litauische, Litewski, Litovskiy, lietuvių",
    'la': "łaciński Latina, Latin",
    'lv': "łotewski Latvian, latviešu",
    'mk': "macedoński Macedonian, Macedonian Slavic, Makedonski, Slavic, македонски",
    'ml': "malajalam Malayalam, Alealum, Malayalani, Malayali, Malean, Maliyad, Mallealle, Mopla, മലയാളം",
    'ms': "malajski Malay, Bahasa Melayu",
    'mt': "maltański Maltese, Malti",
    'mr': "marathi Marathi, Maharashtra, Maharathi, Malhatee, Marthi, Muruthu, मराठी",
    'mn': "mongolski Mongolian, Монгол хэлний бүлэг",
    'nv': "nawaho Navajo, Navaho, Diné bizaad",
    'nap': "neapolitański Neapolitan-Calabrese, Neapolitan",
    'nl': "niderlandzki Dutch, Hollands, Nederlands",
    'de': "niemiecki German, Tedesco, Deutsch",
    'nb': "norweski Norwegian, norsk bokmål",
    'hy': "ormiański Armenian, Armjanski Yazyk, Ena, Ermeni Dili, Ermenice, Somkhuri, հայերեն",
    'fa': "perski Persian, Farsi, فارسی",
    'pl': "polski Polish, Polnisch, Polski",
    'pt': "portugalski Portuguese, Português, português",
    'rom': "romski Romany, Romani chib",
    'ru': "rosyjski Russian, Russki, русский",
    'ro': "rumuński Romanian, Daco-Rumanian, Moldavian, Rumanian, română",
    'sa': "sanskryt Sanskrit, संस्कृतम्",
    'sr': "serbski Serbian, Montenegrin, српски",
    'sk': "słowacki Slovak, Slovakian, Slovencina, slovenčina",
    'sl': "słoweński Slovenian, Slovenscina, slovenščina",
    'sw': "suahili Swahili, Kiswahili",
    'sv': "szwedzki Swedish, Ruotsi, Svenska, svenska",
    'tg': "tadżycki Tajik, Galcha, Tadzhik, Tajiki Persian, Tojiki, Тоҷикӣ",
    'tl': "tagalski Tagalog",
    'th': "tajski Thai, Central Tai, Siamese, Standard Thai, Thaiklang, ไทย",
    'ta': "tamilski Tamil, Damulian, Tamal, Tamalsan, Tambul, Tamili, தமிழ்",
    'tt': "tatarski Tatar, Tartar, Татар теле",
    'te': "telugu Telugu, Andhra, Gentoo, Tailangi, Telangire, Telegu, Telgi, Tengu, Terangi, Tolangan, తెలుగు",
    'tr': "turecki Turkish, Anatolian, Türkisch, Türkçe",
    'tk': "turkmeński Turkmen, Trukhmen, Trukhmeny, Turkmani, Turkmanian, Turkmenler, Turkomans, Türkmen dili",
    'uk': "ukraiński Ukrainian, українська",
    'ur': "urdu Urdu, Bihari, اردو",
    'uz': "uzbecki Uzbek, Oʻzbek tili",
    'vo': "volapuk Volapük",
    'hu': "węgierski Hungarian, Magyar, magyar",
    'vi': "wietnamski Vietnamese, Annamese, Ching, Gin, Jing, Kinh, Viet, Tiếng Việt",
    'it': "włoski Italian, Italiano, italiano",
    'zu': "zulu Isizulu, Zunda, Zulu",
    'oun': "!o!kung !O!ung",
    'nmn': "!Xóõ Ng|amani, Tsasi",
    'alu': "'Are'are Areare",
    'kud': "'Auhelawa Kurada, Nuakata, Ulada, 'Urada",
    'hnh': "//Ani |Anda, Handá, Handádam, Handa-Khwe, Handakwe-Dam, Ts'exa, Ts'éxa",
    'gnk': "//Gana Dxana, G||ana, G||ana-Khwe, Gxana, Gxanna, Kanakhoe",
    'xeg': "//Xegwi ||Xegwe, ||Xekwi, Abathwa, Amabusmana, Amankgqwigqwi, Batwa, Boroa, Bush-C, Gi|kxigwi, Ki||kxigwi, Kloukle, Lxloukxle, Nkqeshe, Tloue, Tloutle",
    'gwj': "/Gwi Dcui, G!wikwe, G|wi, G|wikhwe, Gcwi",
    'xam': "/Xam |Kamka!e, |Kham-Ka-!k'e, |Xam-Ka-!k'e",
    'huc': "=/Hua |Hû, |Hua, ǂHoa, ǂHoã, ǂHoan, ǂHua-Owani",
    'apq': "A-Pucikwar Puchikwar, Pucikwar",
    'aou': "A'ou |Auo, Ayo",
    'aiw': "Aari Aarai, Ara, Ari, Aro, 'Shankilla', 'Shankilligna', 'Shankillinya'",
    'aas': "Aasáx Aasá, Asá, Asak, Asax, Assa, 'Dorobo', Il Konono, Lamanik, 'Ndorobo'",
    'kbt': "Abadi Gabadi, Kabadi",
    'abg': "Abaga Wagaba",
    'abf': "Abai Sungai",
    'abm': "Abanyom Abanyum, Befun, Bofon, Mbofon",
    'mij': "Abar",
    'aau': "Abau Green River",
    'abq': "abazyński Abazin, Abazintsy, Ashuwa, Abaza",
    'ab': "abchaski Abkhazian, Abxazo, Аҧсуа бызшәа",
    'aba': "Abé Abbé, Abbey, Abi",
    'abp': "Abellen Ayta Abenlen, Aburlen Negrito, Ayta Abellen Sambal",
    'abi': "Abidji Abiji",
    'bsa': "Abinomn Avinomen, 'Baso', Foja, Foya",
    'axb': "Abipon Abipones",
    'ash': "Abishira Abigira, Abiquira, Agouisiri, Auishiri, Avirxiri, Ixignor, Tequraca, Vacacocha",
    'aob': "Abom",
    'abo': "Abon Abõ, Abong, Ba'ban",
    'abr': "Abron Bron, Brong, Doma, Gyaman",
    'ado': "Abu Adjora, Adjoria, Azao",
    'aah': "Abu' Arapesh Ua",
    'abn': "Abua Abuan",
    'abz': "Abui 'Barawahing', Barue, Namatalaki",
    'kgr': "Abun A Nden, Karon, Manif, Yimbun",
    'abu': "Abure Abonwa, Abouré, Abule, Akaplass",
    'mgj': "Abureni Mini",
    'tpx': "Acatepec Me'phaa Acatepec Tlapanec, Me'pa, Me'pa Wí'ìn, Me'phaa, Tlapaneco de Acatepec, Tlapaneco del suroeste, Western Tlapanec",
    'ace': "aceh Achinese, Acehnese, Achehnese, Bahsa Acèh",
    'aca': "Achagua Ajagua, Xagua",
    'acn': "Achang Acang, Ach'ang, Achung, Ahchan, Atsang, Maingtha, Mönghsa, Ngacang, Ngac'ang, Ngachang, Ngatsang, Ngo Chang, Ngochang, Xiandao",
    'yif': "Ache",
    'guq': "Aché Ache-Guayaki, Axe, 'Guaiaqui', 'Guayakí', 'Guoyagui'",
    'acz': "Acheron Aceron, Achurun, Asheron, Garme",
    'acr': "Achi",
    'act': "Achterhoeks Aachterhoeks, Achterhoek",
    'acu': "Achuar-Shiwiar Achual, Achuale, Achuar, Achuara, Jivaro, Maina",
    'acv': "Achumawi Achomawi, Pitt River",
    'akv': "achwaski 'Aqwalazul, Ashvado, Axvax, Ghahvalal, Akhvakh",
    'acs': "Acroá Coroá",
    'ach': "aczoli Acooli, Akoli, Atscholi, Dok Acoli, Gang, Log Acoli, Lwo, Lwoo, Shuli, Acoli",
    'adb': "Adabe Ataura, Atauran, Atauro, Atauru, Raklu Un, Raklu-Un",
    'xad': "Adai",
    'fub': "Adamawa Fulfulde Adamawa Fulani, Boulbe, Domona, Dzemay, Eastern Fulani, Foulfoulde, Ful, Fula, Fulata, Fulbe, Fulfulde, Mbororo, Palata, Peul, Peulh",
    'adn': "Adang Alor",
    'adq': "Adangbe Adan, Adantonwi, Agotime, Dangbe",
    'ada': "Adangme",
    'adp': "Adap",
    'kad': "Adara Kadara",
    'tiu': "Adasen Addasen, Addasen Tinguian, Itneg Adasen",
    'ade': "Adele Bedere, Bidire, Gadre, Gidire",
    'adh': "Adhola Dhopadhola, Ludama",
    'adi': "Adi Abhor, Abor, Boga'er Luoba, Lhoba, Luoba",
    'wsg': "Adilabad Gondi",
    'adj': "Adioukrou Adjukru, Adyoukrou, Adyukru, Ajukru",
    'dth': "Adithinngithigh Adetingiti",
    'ort': "Adivasi Oriya Adiwasi Oriya, Kotia Oriya, Kotiya, Tribal Oriya",
    'gas': "Adiwasi Garasia Adiwasi Girasia, Adiwasi Gujarati, Girasia",
    'adt': "Adnyamathanha Ad'n'amadana, Adynyamathanha, Anjimatana, Anjiwatana, Archualda, Benbakanjamata, Binbarnja, Gadjnjamada, Jandali, Kanjimata, Keydnjmarda, Mardala, Nimalda, Nuralda, Umbertana, Unyamootha, Wailbi, Wailpi, Waljbi, Wipie",
    'adr': "Adonara Nusa Tadon, Sagu, Vaiverang, Waiwerang",
    'adu': "Aduge",
    'ady': "adygejski Adygei, Adygey, Circassian, Kiakh, Kjax, Lower Circassian, West Circassian, Adyghe",
    'adz': "Adzera Acira, Atsera, Atzera, Azera",
    'aez': "Aeka Ajeka",
    'awi': "Aekyom Aiwin, Akium, Awin, West Awin",
    'xae': "Aequian",
    'aeq': "Aer",
    'aal': "Afade Afada, Afadeh, Affade, Kotoko, Mogari",
    'aa': "Afar Adal, 'Afar Af, Afaraf, 'Danakil', 'Denkel', Qafar",
    'aft': "Afitti Affitti, Dinik, Ditti, Unietti",
    'afh': "Afrihili",
    'afs': "Afro-Seminole Creole Afro-Seminol Criollo, Afro-Seminole",
    'agd': "Agarabi Agarabe, Bare",
    'agi': "Agariya Agaria, Agharia, Agoria",
    'agc': "Agatu North Idoma, Ochekwu",
    'avo': "Agavotaguerra Agavotokueng, Agavotoqueng",
    'agq': "aghem Wum, Yum, Aghem",
    'ahh': "Aghu Djair, Dyair",
    'ggr': "Aghu Tharnggalu",
    'gtu': "Aghu-Tharnggala Aghu Tharnggala, Aghu Tharnggalai, Aghu Tharnggalu",
    'mis_agk': "Aghul Koshan dialect",
    'aif': "Agi",
    'kit': "Agob Dabu",
    'ibm': "Agoi Ibami, Ro Bambami, Wa Bambani, Wagoi",
    'agu': "Aguacateco Aguacatec",
    'aga': "Aguano Aguanu, Awano, Santa Crucino, Uguano",
    'agr': "Aguaruna Aguajún, Ahuajún, Awajún",
    'agx': "agulski Aghul-ch'al, Agul, Aghul",
    'mis_tok': "toki pona",
}

    # _descriptions = {
    #     CONFIGURATIONAL.LAYOUT_ADJUSTMENT_METHOD_LONG_FLAG: "Replaces language-specific characters with corresponding characters of latin alphabet. 'keyboard'" \
    #                                 " option replaces characters with those of the same placement at default English layout."\
    #                                 " 'native' option replaces characters to those of a corresponding pronunciation (or meaning in case of Chinese).",
    #     CONFIGURATIONAL.LAYOUT_ADJUSTMENT_LANG_LONG_FLAG: "Language to work with the script adjustment",
    # }
    #
    # @classmethod
    # def get_description(cls, flag: str):
    #     try:
    #         return cls._descriptions[flag]
    #     except KeyError:
    #         return ''

# _possible_config_values = {
#     Configs.DEFAULT_TRANSLATIONAL_MODE: [SHORT_FLAGS.SINGLE, SHORT_FLAGS.MULTI_LANG, SHORT_FLAGS.MULTI_WORD,
#                                          FLAGS.SINGLE, FLAGS.MULTI_LANG, FLAGS.MULTI_WORD],
#     Configs.LANG_SPEC_ADJUSTMENT: [LanguageSpecificAdjustmentValues.NONE,
#                                    LanguageSpecificAdjustmentValues.NATIVE,
#                                    LanguageSpecificAdjustmentValues.KEYBOARD],
#     Configs.LANG_LIMIT: 'Any positive number or 0 to cancel the limit out',
#     Configs.ADJUSTMENT_LANG: f'Any language of a different default layout than English or nothing {layout_examples}',
#     Configs.SAVED_LANGS: f'Any language {lang_examples}',
# }

# lang_examples = '(np. en, pl, de, es)'
# layout_examples = '(np. de, uk, ru, zh)'

# def get_possible_values_for(name: str):
#     if name in _possible_config_values:
#         return _possible_config_values[name]
#     return None