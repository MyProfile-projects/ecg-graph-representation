# -*- coding: utf-8 -*-
"""
Checks EVERY reference of the manuscript that has a DOI against Crossref:
  * does the DOI exist,
  * does the title in the manuscript match the title registered for this DOI,
  * do the authors match (first author, number of authors, the 5 + "et al." rule).

    python check_refs.py manuscript.tex

Needs internet (works in Colab). Prints only what needs attention, then a summary.
"""
import json, re, sys, time, unicodedata, urllib.request
from difflib import SequenceMatcher


def norm(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'\\[a-zA-Z]+|[{}\\]', '', s)
    return re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split()


def crossref(doi):
    req = urllib.request.Request('https://api.crossref.org/works/' + doi,
                                 headers={'User-Agent': 'ref-check (mailto:you@example.com)'})
    return json.load(urllib.request.urlopen(req, timeout=30))['message']


def main(path):
    tex = open(path, encoding='utf-8').read()
    items = re.findall(r'\\bibitem\{([^}]*)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', tex, re.S)
    problems, ok, nodoi = 0, 0, []
    for n, (key, txt) in enumerate(items, 1):
        txt = ' '.join(txt.split())
        m = re.search(r'(?:https://doi\.org/|doi:\s*)(\S+?)\.?$', txt)
        if not m:
            nodoi.append(f'[{n}] {key}')
            continue
        doi = m.group(1).replace('\\_', '_')
        parts = txt.split('. ', 1)
        authors, rest = parts[0], parts[1] if len(parts) > 1 else ''
        title = rest.split('. ')[0]
        try:
            msg = crossref(doi)
        except Exception as e:
            print(f'[{n}] {key}: DOI NOT FOUND ({doi}) - {e}'); problems += 1; continue
        cr_title = (msg.get('title') or [''])[0]
        sim = SequenceMatcher(None, ' '.join(norm(title)), ' '.join(norm(cr_title))).ratio()
        people = [a for a in msg.get('author', []) if 'family' in a]
        n_auth = len(people)
        first = norm(people[0]['family'])[0] if people else ''
        ms_first = norm(authors.split(',')[0])[0] if authors else ''
        has_etal = 'et al' in authors
        listed = len([a for a in authors.replace(' et al', '').split(', ') if a.strip()])
        notes = []
        if sim < 0.75:
            notes.append(f'title differs (similarity {sim:.2f}):\n      manuscript: {title}\n      Crossref:   {cr_title}')
        if people and first != ms_first:
            notes.append(f'first author differs: manuscript "{authors.split(",")[0]}", Crossref "{people[0]["family"]}"')
        if n_auth:
            if n_auth > 5 and not (has_etal and listed == 5):
                notes.append(f'{n_auth} authors: should be first 5 + et al. (manuscript lists {listed}{" + et al." if has_etal else ""})')
            if n_auth <= 5 and (has_etal or listed != n_auth):
                notes.append(f'{n_auth} authors: all should be listed without et al. (manuscript lists {listed}{" + et al." if has_etal else ""})')
        if notes:
            problems += 1
            print(f'[{n}] {key}:\n    ' + '\n    '.join(notes))
        else:
            ok += 1
        time.sleep(0.2)
    print(f'\nChecked {ok + problems} references with DOI: {ok} OK, {problems} need attention.')
    print('Without DOI (check by hand):', ', '.join(nodoi))
    print('Note: Crossref sometimes lists organisations as authors or truncates author lists '
          '(e.g. JAMA) - such cases are reported but may be false alarms.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'manuscript.tex')
