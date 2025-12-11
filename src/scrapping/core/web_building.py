def get_default_headers():
    return {
        'User-Agent': f'Scraplang/3.8.1 (piotr10tutek@poczta.onet.pl)',

        'Accept-Language': 	'en-US,en;q=0.5',
        'Accept-Encoding': 	'gzip, deflate, br',
        'Accept': 	'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Referer': 	'http://www.google.com/',
    }

class UrlBuilder:
    MAIN_URL: str = ''
