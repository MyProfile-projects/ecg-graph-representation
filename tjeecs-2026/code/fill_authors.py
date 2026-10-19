# Prints the first five authors (TJEECS style: "Surname AB") for each DOI.
# Run anywhere with internet access (e.g. a new Colab cell):  python fill_authors.py
import json, urllib.request

DOIS = {
    # still missing (needed in manuscript.tex)
    'Chupov 2021':        '10.15826/urej.2021.5.4.001',
    'Karampour 2024':     '10.1109/ICBME64381.2024.10895018',
    'Kiselev 2016':       '10.2174/1874192401610010035',
    'Makarov 2014':       '10.15829/1560-4071-2014-2-6-71',
    'Nilashi 2020':       '10.1007/s40815-020-00828-7',
    'Schmidt 2020':       '10.1038/s41598-020-72034-2',
    # only in manuscript_full.tex
    'Fan 2025':           '10.3390/sym17030449',
    'Yanez-Sepulveda 2024': '10.1080/02673843.2024.2417903',
    # completed from memory (orange [verify]) - confirm
    'Brignole 2018':      '10.1093/eurheartj/ehy037',
    'Cheng 2009':         '10.1001/jama.2009.888',
    'Freeman 2011':       '10.1007/s10286-011-0119-5',
    'Guzzetti 2005':      '10.1161/CIRCULATIONAHA.104.518449',
    'Kligfield 2007':     '10.1161/CIRCULATIONAHA.106.180200',
    'Porta 2001':         '10.1109/10.959324',
    'Virtanen 2020':      '10.1038/s41592-019-0686-2',
}

def initials(given):
    parts = given.replace('-', ' ').replace('.', ' ').split()
    return ''.join(p[0] for p in parts if p)

for name, doi in DOIS.items():
    try:
        req = urllib.request.Request('https://api.crossref.org/works/' + doi,
                                     headers={'User-Agent': 'ref-check (mailto:you@example.com)'})
        msg = json.load(urllib.request.urlopen(req, timeout=20))['message']
        au = msg.get('author', [])
        names = [f"{a.get('family', a.get('name', '?'))} {initials(a.get('given', ''))}".strip()
                 for a in au]
        line = ', '.join(names[:5]) + (' et al.' if len(names) > 5 else '.')
        print(f'{name:22s} ({len(names)} authors): {line}')
    except Exception as e:
        print(f'{name:22s}: not found ({e})')
# No DOI, check by hand: Timofeev 2025 (book), Zavoyskikh 2019 (ITNT-2019 proceedings).
