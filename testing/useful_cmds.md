
Recurl wiktio pages
```bash
lsd | grep wik | cut -d - -f2 | cut -d . -f1 | xargs -I#  curl -L https://en.wiktionary.org/w/api.php -d 'action=parse' -d 'format=json' -d 'prop=text' -d 'page=#' -o wiktio-#.html
```
